"""
Pre-processing and annotating Fisher transcripts using 
a SOTA joint parser and disfluency detector model. For 
a complete description of the model, please refer to 
the following paper:
https://www.aclweb.org/anthology/2020.acl-main.346.pdf


* DisfluencyTagger --> finds disfluency labels
* Parser --> finds constituency parse trees
* Annotate --> pre-processes transcripts for annotation

(c) Paria Jamshid Lou, 14th July 2020.
"""

import sys
sys.path.append("/home/stud-leonardbongard/Bachelor_Arbeit/ba-leonard-bongard/models/submodules/english-fisher-annotations")

import codecs
import fnmatch
import os
import re   
import torch

import parse_nk


class DisfluencyTagger:
    """
    This class is called when self.disfluency==True.    

    Returns:
        A transcript with disfluency labels:
            e.g. "she E she _ likes _ movies _"
            where "E" indicate that the previous 
            word is disfluent and "_" shows that 
            the previous word is fluent.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
 
    @staticmethod
    def fluent(tokens):
        leaves_tags = [t.replace(")","")+" _" for t in tokens if ")" in t]      
        return " ".join(leaves_tags)

    @staticmethod
    def disfluent(tokens):
        # remove first and last brackets
        tokens, tokens[-1] = tokens[1:], tokens[-1][:-1]
        open_bracket, close_bracket, pointer = 0, 0, 0      
        df_region = False
        tags = []
        while pointer < len(tokens):
            open_bracket += tokens[pointer].count("(")                
            close_bracket += tokens[pointer].count(")")
            if "(EDITED" in tokens[pointer]:  
                open_bracket, close_bracket = 1, 0             
                df_region = True
                
            elif ")" in tokens[pointer]:
                label = "E" if df_region else "_"  
                tags.append(
                    (tokens[pointer].replace(")", ""), label)
                    )                 
            if all(
                (close_bracket,
                open_bracket == close_bracket)
                ):
                open_bracket, close_bracket = 0, 0
                df_region = False            

            pointer += 1
        return " ".join(list(map(lambda t: " ".join(t), tags)))


"""
class Parser_OLD(DisfluencyTagger):  # new parser system to load model only ones  
    "" "
    Loads the pre-trained parser model to find silver parse trees     
   
    Returns:
        Parsed and disfluency labelled transcripts
    "" "
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def torch_load(self):
        if parse_nk.use_cuda:
            return torch.load(
                self.model
                )
        else:
            return torch.load(
                self.model, 
                map_location=lambda storage, 
                location: storage,
                )

    def run_parser(self, input_sentences, vocab_path, remove_disfluency_words_bool=False):
        eval_batch_size = 1
        print("Loading model from {}...".format(self.model))  # TODO load only once insetead every time
        assert self.model.endswith(".pt"), "Only pytorch savefiles supported"
        info = self.torch_load()  # TODO load only once insetead every time

        assert "hparams" in info["spec"], "Older savefiles not supported"

        info["spec"]["hparams"]["bert_model"] = vocab_path  # TODO load only once insetead every time

        "" "
        info["spec"]:
        {'hparams': {'attention_dropout': 0.2, 'bert_do_lower_case': True, 'bert_model': './model/bert-base-uncased.tar.gz', 'bert_transliterate': '', 'char_lstm_input_dropout': 0.2, 'clip_grad_norm': 0, 'd_char_emb': 32, 'd_ff': 2048, 'd_kv': 64, 'd_label_hidden': 250, 'd_model': 1024, 'd_tag_hidden': 250, 'elmo_dropout': 0.5, 'embedding_dropout': 0.0, 'learning_rate': 5e-05, 'learning_rate_warmup_steps': 160, 'max_consecutive_decays': 3, 'max_len_dev': 0, 'max_len_train': 0, 'morpho_emb_dropout': 0.2, 'num_heads': 8, 'num_layers': 2, 'num_layers_position_only': 0, 'partitioned': True, 'predict_tags': False, 'relu_dropout': 0.1, 'residual_dropout': 0.2, 'sentence_max_len': 300, 'silver_weight': 4, 'step_decay': True, 'step_decay_factor': 0.5, 'step_decay_patience': 5, 'tag_emb_dropout': 0.2, 'tag_loss_scale': 5.0, 'timing_dropout': 0.0, 'use_bert': True, 'use_bert_only': False, 'use_chars_lstm': False, 'use_elmo': False, 'use_tags': False, 'use_words': False, 'word_emb_dropout': 0.4}, 'char_vocab': <vocabulary.Vocabulary object at 0x7f56a3d144d0>, 'label_vocab': <vocabulary.Vocabulary object at 0x7f56a3c99e90>, 'word_vocab': <vocabulary.Vocabulary object at 0x7f56a3cb1490>, 'tag_vocab': <vocabulary.Vocabulary object at 0x7f56a3314c90>}
        "" "
        parser = parse_nk.NKChartParser.from_spec(
            info["spec"], 
            info["state_dict"],

        )  # TODO load only once insetead every time

        print("Parsing sentences...")
        sentences = [sentence.split() for sentence in input_sentences]
        # Tags are not available when parsing from raw text, so use a dummy tag
        if "UNK" in parser.tag_vocab.indices:
            dummy_tag = "UNK"
        else:
            dummy_tag = parser.tag_vocab.value(0)
        
        all_predicted = []
        for start_index in range(0, len(sentences), eval_batch_size):
            subbatch_sentences = sentences[start_index:start_index+eval_batch_size]
            subbatch_sentences = [[(dummy_tag, word) for word in sentence] for sentence in subbatch_sentences]
            predicted, _ = parser.parse_batch(subbatch_sentences)
            del _
            all_predicted.extend([p.convert() for p in predicted])
        
        parse_trees, df_labels = [], []
        for tree in all_predicted:      
            linear_tree = tree.linearize()

            parse_trees.append(linear_tree)
            if self.disfluency:
                tokens = linear_tree.split()
                # disfluencies are dominated by EDITED nodes in parse trees
                if "EDITED" not in linear_tree: 
                    df_labels.append(self.fluent(tokens))
                elif not remove_disfluency_words_bool:
                    df_labels.append(self.disfluent(tokens))
                    
        return parse_trees, df_labels
"""

class Parser(DisfluencyTagger):
    """
    Loads the pre-trained parser model to find silver parse trees     
   
    Returns:
        Parsed and disfluency labelled transcripts
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        

    def torch_load(self):
        if parse_nk.use_cuda:
            return torch.load(
                self.model
                )
        else:
            return torch.load(
                self.model, 
                map_location=lambda storage, 
                location: storage,
                )

    def run_parser(self, input_sentences, remove_disfluency_words_bool=False):
        eval_batch_size = 1
                

        """
        info["spec"]:
        {'hparams': {'attention_dropout': 0.2, 'bert_do_lower_case': True, 'bert_model': './model/bert-base-uncased.tar.gz', 'bert_transliterate': '', 'char_lstm_input_dropout': 0.2, 'clip_grad_norm': 0, 'd_char_emb': 32, 'd_ff': 2048, 'd_kv': 64, 'd_label_hidden': 250, 'd_model': 1024, 'd_tag_hidden': 250, 'elmo_dropout': 0.5, 'embedding_dropout': 0.0, 'learning_rate': 5e-05, 'learning_rate_warmup_steps': 160, 'max_consecutive_decays': 3, 'max_len_dev': 0, 'max_len_train': 0, 'morpho_emb_dropout': 0.2, 'num_heads': 8, 'num_layers': 2, 'num_layers_position_only': 0, 'partitioned': True, 'predict_tags': False, 'relu_dropout': 0.1, 'residual_dropout': 0.2, 'sentence_max_len': 300, 'silver_weight': 4, 'step_decay': True, 'step_decay_factor': 0.5, 'step_decay_patience': 5, 'tag_emb_dropout': 0.2, 'tag_loss_scale': 5.0, 'timing_dropout': 0.0, 'use_bert': True, 'use_bert_only': False, 'use_chars_lstm': False, 'use_elmo': False, 'use_tags': False, 'use_words': False, 'word_emb_dropout': 0.4}, 'char_vocab': <vocabulary.Vocabulary object at 0x7f56a3d144d0>, 'label_vocab': <vocabulary.Vocabulary object at 0x7f56a3c99e90>, 'word_vocab': <vocabulary.Vocabulary object at 0x7f56a3cb1490>, 'tag_vocab': <vocabulary.Vocabulary object at 0x7f56a3314c90>}
        """
        

        print("Parsing sentences...")
        sentences = [sentence.split() for sentence in input_sentences]
        # Tags are not available when parsing from raw text, so use a dummy tag
        if "UNK" in self.parser.tag_vocab.indices:
            dummy_tag = "UNK"
        else:
            dummy_tag = self.parser.tag_vocab.value(0)
        
        all_predicted = []
        for start_index in range(0, len(sentences), eval_batch_size):
            subbatch_sentences = sentences[start_index:start_index+eval_batch_size]
            subbatch_sentences = [[(dummy_tag, word) for word in sentence] for sentence in subbatch_sentences]
            predicted, _ = self.parser.parse_batch(subbatch_sentences)
            del _
            all_predicted.extend([p.convert() for p in predicted]) 
        
        parse_trees, df_labels = [], []
        print([p.convert() for p in predicted])
        print("allp", all_predicted)
        for tree in all_predicted:      
            print("tree", tree)
            linear_tree = tree.linearize()

            parse_trees.append(linear_tree)
            if self.disfluency:
                tokens = linear_tree.split()
                print("tokens", tokens)
                print("linear tree", linear_tree)
                # disfluencies are dominated by EDITED nodes in parse trees
                if "EDITED" not in linear_tree: 
                    df_labels.append(self.fluent(tokens))
                
                elif not remove_disfluency_words_bool:
                    df_labels.append(self.disfluent(tokens))
                print(df_labels)
                print(parse_trees)
                    
        return parse_trees, df_labels

           
class Annotate(Parser):   
    """
    Writes parsed and disfluency labelled transcripts into 
    *_parse.txt and *_dys.txt files, respectively.

    """ 
    def __init__(self, **kwargs):
        self.input_path = kwargs["input_path"]
        self.output_path = kwargs["output_path"] 
        self.model = kwargs["model"] 
        self.disfluency = kwargs["disfluency"] 
        self.remove_df= kwargs["remove_df_words"]
        self.vocab_path = kwargs["vocab_path"]

        print("Loading model from {}...".format(self.model)) 
        assert self.model.endswith(".pt"), "Only pytorch savefiles supported"
        self.info = self.torch_load()

        assert "hparams" in self.info["spec"], "Older savefiles not supported"

        self.info["spec"]["hparams"]["bert_model"] = kwargs["vocab_path"]

        self.parser = parse_nk.NKChartParser.from_spec(
            self.info["spec"], 
            self.info["state_dict"])


    def setup(self): 
        # all_2004 = self.parse_sentences(            
        #     trans_data=os.path.join(
        #         "LDC2004T19", 
        #         "fe_03_p1_tran", 
        #         "data", 
        #         "trans",
        #         ),
        #     parsed_data="fisher-2004-annotations"
        # )

        # all_2005 = self.parse_sentences(

        #     parsed_data="fisher-2005-annotations"
        # )
        return self.parse_sentences_for_one_file(self.remove_df)


    def parse_sentences_for_one_file(self, remove_df_words): 
        # Loop over transcription files
        trans_file =  self.input_path
        segments = self.read_transcription(trans_file) 
        print("segments:", segments)
        # Loop over cleaned/pre-proceesed transcripts         
        doc = [segment for segment in segments if segment]    
        parse_trees, df_labels = self.run_parser(doc, remove_df_words)
        df_labels = self.remove_labels(df_labels)

        # if self.disfluency:
        #     print("Write into file:", self.output_path)
        #     with open(self.output_path, "w") as output_file:
        #         output_file.write("\n".join(df_labels))

        return "\n".join(df_labels)

    def remove_labels(self, df_labels):
        """
        Method to remove all disfluency annotated labels
        """
        return [label.replace(" _", "") for label in df_labels]

    def parse_sentences(self, trans_data, parsed_data):
        #input_dir = os.path.join(self.input_path, trans_data)
        input_dir = os.path.join(self.input_path)
        output_dir = os.path.join(self.output_path, parsed_data)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)   
        # Loop over transcription files
        print(input_dir)
        for root, dirnames, filenames in os.walk(input_dir):
            print(root, dirnames, filenames )
            for filename in fnmatch.filter(filenames, "*.txt"):
                print(filename)
                trans_file = os.path.join(root, filename)
                segments = self.read_transcription(trans_file) 
                # Loop over cleaned/pre-proceesed transcripts         
                doc = [segment for segment in segments if segment]    
                parse_trees, df_labels = self.run_parser(doc)
                # Write constituency parse trees and disfluency labels into files
                new_filename = os.path.join(
                    output_dir, 
                    os.path.basename(trans_file[:-4])+"_parse.txt"
                    )
                with open(new_filename, "w") as output_file:
                    output_file.write("\n".join(parse_trees))

                if self.disfluency:
                    new_filename = os.path.join(
                        output_dir, 
                        os.path.basename(trans_file[:-4])+"_dys.txt"
                        )
                    with open(new_filename, "w") as output_file:
                        output_file.write("\n".join(df_labels))

        return

    def read_transcription(self, trans_file, skip_token=False):  # Skip token isnerted by leonard
        skip = 3 if skip_token else 0
        with codecs.open(trans_file, "r", "utf-8") as fp:
            for line in fp:
                if line.startswith("#") or len(line) <= 1:
                    continue                
                tokens = line.split() 

                yield " ".join(tokens[skip:])
                #yield self.validate_transcription(
                 #   " ".join(tokens[skip:])
                 #   )           

    @staticmethod
    def validate_transcription(label):
        if re.search(r"[0-9]|[(<\[\]&*{]", label):
            return None

        label = label.replace("_", " ")
        label = re.sub("[ ]{2,}", " ", label)
        label = label.replace(".", "")
        label = label.replace(",", "")
        label = label.replace(";", "")
        label = label.replace("?", "")
        label = label.replace("!", "")
        label = label.replace(":", "")
        label = label.replace("\"", "")
        label = label.replace("'re", " 're")
        label = label.replace("'ve", " 've")
        label = label.replace("n't", " n't")
        label = label.replace("'ll", " 'll")
        label = label.replace("'d", " 'd")
        label = label.replace("'m", " 'm")
        label = label.replace("'s", " 's")
        label = label.strip()
        label = label.lower()

        return label if label else None   
