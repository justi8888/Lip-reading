import torch
import torchaudio
from cosine import WarmupCosineScheduler
from datamodule.transforms import TextTransform

from pytorch_lightning import LightningModule
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.scorers.ctc import CTCPrefixScorer
from torchviz import make_dot
import char_lists
import csv


def append_to_csv(file_path, data):
    with open(file_path, 'a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["wer", "cer", "bleu", "hypothesis", "reference"])
        
        if csv_file.tell() == 0:  # If file is empty, write header
            writer.writeheader()
        writer.writerow(data)

def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())

def compute_character_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower(), seq2.lower())
    

class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.backbone_args = self.cfg.model
        self.wer = []
        self.cer = []
        self.output_type = self.cfg.output_type
        self.csv_file = self.cfg.result_csv
        
        if self.output_type == 'char_nod':
            self.token_list = char_lists.char_nod_list
        elif self.output_type == 'char':
            self.token_list = char_lists.char_list
        elif self.output_type == 'token':
            self.text_transform = TextTransform()
            self.token_list = self.text_transform.token_list
        else:
            raise NotImplementedError("Possible output types are char, char_nod and token. Please change it in configuration file.")
        
        
        if self.cfg.testing:
            self.model = E2E(len(self.token_list), self.backbone_args)
            
        # training from pre-trained english model    
        else:
            if self.cfg.pretrained_model_path:
                ckpt = torch.load(self.cfg.pretrained_model_path, map_location=lambda storage, loc: storage)
                original_output_size = ckpt['decoder.output_layer.weight'].size()[0]
                self.model = E2E(original_output_size, self.backbone_args)
                
                if self.output_type == 'char_nod' and self.cfg.model_type == 'simple':
                    self.model.load_state_dict(ckpt, strict=True)
                elif self.cfg.continue_training:
                    self.model.load_state_dict(ckpt, strict=True)
                    for param in self.model.encoder.frontend.parameters():
                        param.requires_grad = False
                else:  
                    encoder = {k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
                    decoder = {k.replace("decoder.", ""): v for k, v in ckpt.items() if k.startswith("decoder.")}
                    ctc = {k.split('ctc.')[1]: v for k, v in ckpt.items() if k.startswith("ctc")}
                    self.model.encoder.load_state_dict(encoder, strict=True)
                    self.model.decoder.load_state_dict(decoder, strict=True)
                    self.model.ctc.load_state_dict(ctc, strict=True)
                    
                    output_idim = decoder['output_layer.weight'].size()[1]
                    embed_size = decoder['embed.0.weight'].size()[1]
                    
                    print(decoder['output_layer.weight'].size())
                    print(decoder['embed.0.weight'].size())
                    
                    torch.manual_seed(3)
                    # changing required dimensions
                    self.model.decoder.embed[0] = torch.nn.Embedding(len(self.token_list), embed_size)
                    self.model.decoder.output_layer = torch.nn.Linear(output_idim, len(self.token_list))
                    self.model.ctc.ctc_lo = torch.nn.Linear(output_idim, len(self.token_list))
                    
                    self.model.sos = len(self.token_list) - 1
                    self.model.eos = len(self.token_list) - 1
                    self.model.odim = len(self.token_list)
                    self.model.criterion.size = len(self.token_list)
                    self.model.decoder.odim = len(self.token_list)
                    

                if self.cfg.reset_last_layer:
                    torch.manual_seed(3)
                    torch.nn.init.normal_(self.model.decoder.output_layer.weight, mean=0, std=0.01)
                    torch.nn.init.normal_(self.model.decoder.output_layer.bias, mean=0, std=0.01)
                if self.cfg.freeze_frontend:
                    for param in self.model.encoder.frontend.parameters():
                        param.requires_grad = False
                if self.cfg.freeze_encoder:
                    for param in self.model.encoder.parameters():
                        param.requires_grad = False
                
            else:
                self.model = E2E(len(self.token_list), self.backbone_args)

          

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([{"name": "model", "params": self.model.parameters(), "lr": self.cfg.optimizer.lr}], weight_decay=self.cfg.optimizer.weight_decay, betas=(0.9, 0.98))
        scheduler = WarmupCosineScheduler(optimizer, self.cfg.optimizer.warmup_epochs, self.cfg.trainer.max_epochs, len(self.trainer.datamodule.train_dataloader()))
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def forward(self, sample):
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        enc_feat, _ = self.model.encoder(sample.unsqueeze(0).to(self.device), None)
        enc_feat = enc_feat.squeeze(0)

        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        
        if self.output_type == 'char' or self.output_type == 'char_nod':
            hypothesis = parse_hypothesis(nbest_hyps, self.token_list)
            hypothesis = hypothesis.replace("▁", " ").strip()
            return hypothesis.replace("<eos>", "")
        else:
            hypothesis_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
            hypothesis = self.text_transform.post_process(hypothesis_token_id).replace("<eos>", "")
            return hypothesis

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")

    def validation_step(self, batch, batch_idx):
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        for i in range(batch["inputs"].size(0)):
            if i%5 == 0:
                enc_feat, _ = self.model.encoder(batch["inputs"][i].unsqueeze(0).to(self.device), None)
                enc_feat = enc_feat.squeeze(0)
                nbest_hyps = self.beam_search(enc_feat)
                nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
                
                if self.output_type == 'char' or self.output_type == 'char_nod':
                    hypothesis = parse_hypothesis(nbest_hyps, self.token_list)
                    hypothesis = hypothesis.replace("▁", " ").strip()
                    hypothesis = hypothesis.replace("<eos>", "")
                else:
                    hypothesis_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
                    hypothesis = self.text_transform.post_process(hypothesis_token_id).replace("<eos>", "")
                    
                token_id = batch["targets"][i]

                
                if self.output_type == 'char' or self.output_type == 'char_nod':
                    reference = decode_tokens(token_id, self.token_list)
                else:
                    reference = self.text_transform.post_process(token_id)
                    
                
                word_edit_distance = compute_word_level_distance(reference, hypothesis)
                word_length = len(reference.split())
                self.wer.append(word_edit_distance/word_length)
        
                char_edit_distance = compute_character_level_distance(reference, hypothesis)
                char_length = len(reference)
                self.cer.append(char_edit_distance/char_length)
        return self._step(batch, batch_idx, step_type="val")

    def test_step(self, sample, sample_idx):
        enc_feat, _ = self.model.encoder(sample["input"].unsqueeze(0).to(self.device), None)
        enc_feat = enc_feat.squeeze(0)
        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        
        if self.output_type == 'char' or self.output_type == 'char_nod':
            hypothesis = parse_hypothesis(nbest_hyps, self.token_list)
            hypothesis = hypothesis.replace("▁", " ").strip()
            hypothesis = hypothesis.replace("<eos>", "")
        else:
            hypothesis_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
            hypothesis = self.text_transform.post_process(hypothesis_token_id).replace("<eos>", "")

        token_id = sample["target"]
        if self.output_type == 'char' or self.output_type == 'char_nod':
            reference = decode_tokens(token_id, self.token_list)
        else:
            reference = self.text_transform.post_process(token_id)


        word_edit_distance = compute_word_level_distance(reference, hypothesis)
        word_length = len(reference.split())
        wer = word_edit_distance/word_length
        self.wer.append(wer)
        
        char_edit_distance = compute_character_level_distance(reference, hypothesis)
        char_length = len(reference)
        cer = char_edit_distance/char_length
        self.cer.append(cer)
        data_point = {
            "wer": wer, "cer": cer, "hypothesis": hypothesis, "reference": reference
        }
        
        append_to_csv(self.csv_file, data_point)
        return

    def _step(self, batch, batch_idx, step_type):
        loss, loss_ctc, loss_att, acc = self.model(batch["inputs"], batch["input_lengths"], batch["targets"])
        batch_size = len(batch["inputs"])

        if step_type == "train":
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("loss_ctc", loss_ctc, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("loss_att", loss_att, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size)
        else:
            self.log("loss_val", loss, batch_size=batch_size)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size)
            self.log("loss_att_val", loss_att, batch_size=batch_size)
            self.log("decoder_acc_val", acc, batch_size=batch_size)

        if step_type == "train":
            self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.loaders.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()

    def on_test_epoch_start(self):
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)

    def on_test_epoch_end(self):
        self.log("wer", sum(self.wer) / len(self.wer))
        self.log("cer", sum(self.cer) / len(self.cer))
        self.wer = []
        self.cer = []
        
    def on_validation_epoch_end(self):
        self.log("wer", sum(self.wer) / len(self.wer))
        self.log("cer", sum(self.cer) / len(self.cer))
        self.wer = []
        self.cer = []


def get_beam_search_decoder(model, token_list, ctc_weight=0.1, beam_size=40):
    scorers = {
        "decoder": model.decoder,
        "ctc": CTCPrefixScorer(model.ctc, model.eos),
        "length_bonus": LengthBonus(len(token_list)),
        "lm": None
    }

    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": 0.0,
        "length_bonus": 0.0,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )


def decode_tokens(tokens, char_list):
    string_list = tokens.flatten().tolist()
    
    tokenid_as_list = list(map(int, string_list))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace("<space>", " ").replace("<eos>", "")
    return text


def parse_hypothesis(nbest_hyps, char_list):
    # remove sos and get results
    hyp = nbest_hyps[0]
    tokenid_as_list = list(map(int, hyp["yseq"][1:]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    score = float(hyp["score"])

    # convert to string
    text = "".join(token_as_list).replace("<space>", " ")

    return text


