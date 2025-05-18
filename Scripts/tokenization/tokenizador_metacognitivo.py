from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from typing import Optional, List, Dict, Tuple
import json
import os
import sys

# Añadir rutas para importar módulos del directorio core
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Importar componentes del sistema metacognitivo si es necesario
try:
    from core.cerebro_autonomo import CerebroAutonomo
except ImportError:
    # Solo importamos para referencia, no es crítico para el tokenizador
    pass

class MetacognitivoTokenizer:
    """
    Tokenizador personalizado para el sistema metacognitivo en español.
    Basado en BPE con soporte para tokens especiales metacognitivos.
    """
    
    def __init__(self, vocab_file: str, merges_file: str, **kwargs):
        """
        Inicializa el tokenizador metacognitivo.
        
        Args:
            vocab_file: Ruta al archivo de vocabulario (vocab.json)
            merges_file: Ruta al archivo de fusiones (merges.txt)
        """
        # Configuración de tokens especiales
        self.special_tokens = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "additional_special_tokens": [
                "<metacog>", "</metacog>",
                "<reflect>", "<introspect>"
            ]
        }
        
        # Cargar el tokenizador base
        self.tokenizer = Tokenizer(models.BPE.from_files(
            vocab_file=vocab_file,
            merges_file=merges_file,
            unk_token=self.special_tokens["unk_token"]
        ))
        
        # Configurar pre-tokenizador para español
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        
        # Configurar decodificador
        self.tokenizer.decoder = decoders.ByteLevel()
        
        # Configurar post-procesador
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.special_tokens['bos_token']} $A {self.special_tokens['eos_token']}",
            pair=f"{self.special_tokens['bos_token']} $A {self.special_tokens['eos_token']} $B:1 {self.special_tokens['eos_token']}:1",
            special_tokens=[
                (self.special_tokens['bos_token'], 0),
                (self.special_tokens['eos_token'], 1),
            ]
        )
        
        # Configurar tokenizador de HuggingFace
        self.hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            **self.special_tokens,
            model_max_length=1024,
            padding_side="right",
            truncation_side="right"
        )
    
    def __call__(self, *args, **kwargs):
        """Delega la llamada al tokenizador de HuggingFace"""
        return self.hf_tokenizer(*args, **kwargs)
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """Guarda el tokenizador en el directorio especificado"""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Guardar configuración
        config = {
            "model_type": "metacognitivo_espanol",
            "tokenizer_class": "MetacognitivoTokenizer",
            **self.special_tokens
        }
        
        with open(os.path.join(save_directory, "tokenizer_config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # Guardar tokenizador
        self.hf_tokenizer.save_pretrained(save_directory)

# Registrar la clase para que pueda ser encontrada por AutoTokenizer
AutoTokenizer.register("metacognitivo_espanol", MetacognitivoTokenizer)
