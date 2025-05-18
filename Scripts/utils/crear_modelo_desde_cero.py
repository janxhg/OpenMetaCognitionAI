#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo para crear modelos y tokenizadores desde cero para el sistema metacognitivo integrado.
Este script proporciona funciones para generar nuevos modelos sin usar pesos pre-entrenados.
"""

import os
import logging
import torch
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    PreTrainedTokenizerFast, 
    GPT2Config, 
    GPT2LMHeadModel,
    GPT2Tokenizer
)

# Añadir rutas para importar módulos del directorio core
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Importar componentes del sistema metacognitivo si es necesario
try:
    from core.cerebro_autonomo import CerebroAutonomo
    from tokenization.tokenizador_metacognitivo import MetacognitivoTokenizer
except ImportError as e:
    print(f"Nota: No se pudieron importar algunos componentes: {e}")
    print("Esto puede ser normal si solo se está usando este módulo de forma independiente")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def entrenar_tokenizador_desde_cero(textos: List[str], vocab_size: int = 15000, dir_salida: str = None) -> PreTrainedTokenizerFast:
    """
    Entrena un tokenizador desde cero usando textos de ejemplo.
    
    Args:
        textos: Lista de textos para entrenar el tokenizador
        vocab_size: Tamaño del vocabulario
        dir_salida: Directorio donde guardar el tokenizador
        
    Returns:
        Tokenizador entrenado y listo para usar
    """
    # Limitar la cantidad de textos para evitar problemas de memoria
    max_textos = 20  # Reducir a un número muy pequeño para evitar problemas de memoria
    if len(textos) > max_textos:
        logger.info(f"Limitando a {max_textos} textos para entrenamiento de tokenizador (de {len(textos)} disponibles)")
        # Seleccionar textos más cortos
        textos = sorted(textos, key=len)[:max_textos]
    
    logger.info(f"Entrenando tokenizador desde cero con {len(textos)} textos (vocab size: {vocab_size})")
    
    # Crear directorio temporal para archivos de entrenamiento
    tmp_dir = os.path.join(dir_salida, "tmp") if dir_salida else "./tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Tamaño máximo de texto para evitar problemas de memoria
    max_texto_length = 20000  # Reducido a 20KB para minimizar uso de memoria
    
    # Guardar textos en archivos para el entrenamiento (con límite de tamaño)
    file_paths = []
    for i, texto in enumerate(textos):
        if texto.strip():  # Ignorar textos vacíos
            # Truncar textos muy grandes
            if len(texto) > max_texto_length:
                texto = texto[:max_texto_length]
                
            try:
                file_path = os.path.join(tmp_dir, f"texto_{i}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(texto)
                file_paths.append(file_path)
            except Exception as e:
                logger.warning(f"Error al guardar texto {i}: {str(e)}")
    
    if not file_paths:
        raise ValueError("No hay textos válidos para entrenar el tokenizador")
    
    # Crear y entrenar el tokenizador
    # Usar un enfoque incremental para entrenar el tokenizador con menos memoria
    tokenizer = ByteLevelBPETokenizer()
    
    # Entrenar con lotes pequeños de archivos para evitar problemas de memoria
    batch_size = 5  # Procesar solo 5 archivos a la vez
    for i in range(0, len(file_paths), batch_size):
        batch_files = file_paths[i:i+batch_size]
        if i == 0:
            # Primera iteración - entrenar desde cero
            tokenizer.train(
                files=batch_files,
                vocab_size=vocab_size,
                min_frequency=2,
                special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
            )
            logger.info(f"Entrenado tokenizador con lote inicial (archivos {i+1}-{i+len(batch_files)})")
        else:
            # Las siguientes iteraciones - continuar entrenamiento
            try:
                # Intentar continuar entrenamiento
                tokenizer.train(
                    files=batch_files,
                    vocab_size=vocab_size,
                    min_frequency=2
                )
                logger.info(f"Enriquecido tokenizador con lote adicional (archivos {i+1}-{i+len(batch_files)})")
            except Exception as e:
                logger.warning(f"Error al entrenar lote adicional: {str(e)}. Continuando con vocabulario actual.")
                break
    
    # Guardar tokenizador en formato compatible con transformers
    if dir_salida:
        tokenizer_dir = os.path.join(dir_salida, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer.save_model(tokenizer_dir)
        tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
        
        # Crear tokenizer_config.json
        import json
        with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "model_type": "gpt2",
                "add_prefix_space": True,
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "mask_token": "<mask>"
            }, f, indent=2)
        
        # Convertir a tokenizador de HuggingFace
        logger.info(f"Tokenizador guardado en {tokenizer_dir}")
        return PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    
    return None

def crear_modelo_pequeño(config: Dict = None) -> GPT2LMHeadModel:
    """
    Crea un modelo pequeño tipo GPT2 con pesos inicializados aleatoriamente.
    
    Args:
        config: Configuración personalizada del modelo
        
    Returns:
        Modelo inicializado
    """
    # Configuración por defecto para un modelo muy pequeño (memoria limitada)
    config_por_defecto = {
        "vocab_size": 8000,  # Reducido para ahorrar memoria
        "n_positions": 256, # Secuencias más cortas
        "n_ctx": 256,       # Contexto más pequeño
        "n_embd": 192,      # Menos dimensiones de incrustación
        "n_layer": 4,       # Menos capas
        "n_head": 4,        # Menos cabezas de atención
        "bos_token_id": 0,
        "eos_token_id": 2,
    }
    
    # Combinar con configuración personalizada si se proporciona
    if config:
        config_por_defecto.update(config)
    
    # Crear configuración y modelo
    modelo_config = GPT2Config(**config_por_defecto)
    modelo = GPT2LMHeadModel(modelo_config)
    logger.info(f"Modelo pequeño creado con {modelo.num_parameters():,} parámetros")
    
    return modelo

def crear_modelo_y_tokenizador_nuevos(
    textos: List[str],
    dir_salida: str,
    config_modelo: Dict = None,
    vocab_size: int = 15000,
) -> Tuple[GPT2LMHeadModel, PreTrainedTokenizerFast]:
    """
    Crea un modelo y tokenizador completamente nuevos a partir de textos.
    
    Args:
        textos: Lista de textos para entrenar el tokenizador
        dir_salida: Directorio donde guardar el modelo y tokenizador
        config_modelo: Configuración personalizada del modelo
        vocab_size: Tamaño del vocabulario
        
    Returns:
        Tupla (modelo, tokenizador)
    """
    logger.info(f"Creando modelo y tokenizador nuevos en {dir_salida}")
    os.makedirs(dir_salida, exist_ok=True)
    
    # 1. Entrenar tokenizador desde cero
    tokenizer = entrenar_tokenizador_desde_cero(textos, vocab_size, dir_salida)
    if not tokenizer:
        raise ValueError("No se pudo crear el tokenizador")
    
    # 2. Actualizar config con el tamaño de vocabulario real
    if config_modelo is None:
        config_modelo = {}
    config_modelo["vocab_size"] = tokenizer.vocab_size
    
    # 3. Crear modelo pequeño
    modelo = crear_modelo_pequeño(config_modelo)
    
    # 4. Guardar modelo
    modelo_dir = os.path.join(dir_salida, "modelo")
    os.makedirs(modelo_dir, exist_ok=True)
    modelo.save_pretrained(modelo_dir)
    tokenizer.save_pretrained(modelo_dir)
    
    logger.info(f"Modelo y tokenizador creados y guardados en {modelo_dir}")
    return modelo, tokenizer

def extraer_textos_de_materiales(materiales: List[Dict]) -> List[str]:
    """
    Extrae textos de una lista de materiales.
    
    Args:
        materiales: Lista de diccionarios con materiales de estudio
        
    Returns:
        Lista de textos extraídos
    """
    textos = []
    
    # Máximo número de materiales a procesar (para evitar problemas de memoria)
    max_materiales = 40
    if len(materiales) > max_materiales:
        logger.warning(f"Limitando procesamiento a {max_materiales} materiales de {len(materiales)} disponibles")
        materiales = materiales[:max_materiales]
    
    for material in materiales:
        try:
            if isinstance(material, dict) and "contenido" in material:
                # Limitar tamaño de contenido para evitar problemas de memoria
                contenido = material["contenido"]
                if isinstance(contenido, str) and contenido.strip():
                    # Limitar longitud para evitar problemas de memoria
                    if len(contenido) > 20000:  # Reducido a 20KB máximo
                        contenido = contenido[:20000]
                    textos.append(contenido)
            elif isinstance(material, str) and material.strip():
                # Limitar longitud para evitar problemas de memoria
                if len(material) > 20000:  # Reducido a 20KB máximo
                    material = material[:20000]
                textos.append(material)
        except Exception as e:
            logger.warning(f"Error al procesar material: {str(e)}")
            
    logger.info(f"Extraídos {len(textos)} textos válidos de {len(materiales)} materiales")
    return textos

if __name__ == "__main__":
    # Ejemplo básico de uso
    textos_ejemplo = [
        "Este es un texto de ejemplo para entrenar un tokenizador desde cero.",
        "El sistema metacognitivo permite que el modelo observe su propio proceso de aprendizaje.",
        "La introspección y la auto-modificación son capacidades clave de este sistema."
    ]
    
    modelo, tokenizer = crear_modelo_y_tokenizador_nuevos(
        textos_ejemplo,
        "./modelo_nuevo",
        vocab_size=2000  # Pequeño para este ejemplo
    )
    
    print(f"Tokenizador creado con {tokenizer.vocab_size} tokens")
    print(f"Modelo creado con {modelo.num_parameters():,} parámetros")
