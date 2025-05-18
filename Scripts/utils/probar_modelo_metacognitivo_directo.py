#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Probar Modelo Metacognitivo Directo

Este script permite probar un modelo entrenado con el enfoque metacognitivo directo,
que reemplaza las épocas tradicionales por un entrenamiento controlado metacognitivamente.
"""

import os
import sys
import time
import torch
import logging
import argparse
from typing import List, Optional
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AutoTokenizer

# La configuración de logging se hará en main() para permitir guardar en archivo
logger = logging.getLogger(__name__)


def probar_generacion(modelo, tokenizer, prompt: str, max_length: int = 100, 
                     temperatura: float = 0.8, num_secuencias: int = 1,
                     max_tokens: int = 200, device: str = "cuda"):
    """
    Prueba la generación de texto con el modelo metacognitivo directo.
    
    Args:
        modelo: Modelo a probar
        tokenizer: Tokenizador asociado al modelo
        prompt: Texto inicial para generación
        max_length: Longitud máxima de salida
        temperatura: Temperatura para sampling (más alta = más aleatorio)
        num_secuencias: Número de secuencias a generar
        max_tokens: Máximo número de tokens a generar
        device: Dispositivo donde ejecutar el modelo
        
    Returns:
        Lista de textos generados
    """
    logger.info(f"Generando texto a partir de: '{prompt}'")
    logger.info(f"Usando temperatura: {temperatura}")
    
    # Asegurar que se utilizan los tokens correctos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Tokenizar el prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Obtener la longitud máxima del modelo para evitar errores de índice
    try:
        max_model_length = modelo.config.n_positions
        logger.info(f"El modelo tiene una longitud máxima de contexto de {max_model_length}")
        
        # Ajustar max_length para que no exceda la capacidad del modelo
        safe_max_length = min(max_length, max_model_length - 5)  # 5 de margen de seguridad
        safe_max_tokens = min(max_tokens, max_model_length - len(inputs.input_ids[0]) - 5)
        
        if safe_max_length < max_length or safe_max_tokens < max_tokens:
            logger.warning(f"Ajustando parámetros de generación al límite del modelo: {max_model_length}")
            logger.warning(f"max_length ajustado de {max_length} a {safe_max_length}")
            logger.warning(f"max_new_tokens ajustado de {max_tokens} a {safe_max_tokens}")
    except Exception as e:
        logger.warning(f"No se pudo determinar la longitud máxima del modelo: {str(e)}")
        safe_max_length = 50  # Valor seguro por defecto
        safe_max_tokens = 30  # Valor seguro por defecto
    
    # Configurar parámetros de generación
    gen_kwargs = {
        "max_length": safe_max_length,
        "temperature": temperatura,
        "num_return_sequences": num_secuencias,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "top_p": 0.9,
        "top_k": 40,
        "max_new_tokens": safe_max_tokens,
    }
    
    # Generar texto
    try:
        inicio = time.time()
        outputs = modelo.generate(**inputs, **gen_kwargs)
        duracion = time.time() - inicio
        logger.info(f"Generación completada en {duracion:.2f} segundos")
        
        # Decodificar las salidas generadas
        textos_generados = []
        for i, output in enumerate(outputs):
            texto = tokenizer.decode(output, skip_special_tokens=True)
            textos_generados.append(texto)
            logger.info(f"\nSecuencia {i+1}:\n{'-'*40}\n{texto}\n{'-'*40}")
            
        return textos_generados
    
    except Exception as e:
        logger.error(f"Error en generación: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return [f"Error: {str(e)}"]


def cargar_modelo(ruta_modelo: str, device: str = "cuda") -> tuple:
    """
    Carga un modelo metacognitivo directo desde su directorio.
    
    Args:
        ruta_modelo: Ruta al directorio donde está guardado el modelo
        device: Dispositivo donde cargar el modelo
        
    Returns:
        Tupla (modelo, tokenizer)
    """
    try:
        logger.info(f"Cargando modelo desde: {ruta_modelo}")
        
        # Verificar si existe la configuración
        if not os.path.exists(os.path.join(ruta_modelo, "config.json")):
            logger.error(f"No se encontró config.json en {ruta_modelo}")
            return None, None
        
        # Intentar cargar el modelo
        modelo = GPT2LMHeadModel.from_pretrained(ruta_modelo)
        modelo.to(device)
        logger.info(f"Modelo cargado y movido a {device}")
        
        # Intentar cargar tokenizador desde el mismo checkpoint del modelo
        # Este enfoque funcionará si el tokenizador se guardó junto con el modelo
        logger.info(f"Intentando cargar tokenizador desde el mismo checkpoint: {ruta_modelo}")
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(ruta_modelo)
            logger.info("Tokenizador encontrado y cargado desde el checkpoint del modelo")
        except Exception as e:
            logger.warning(f"No se pudo cargar el tokenizador desde el checkpoint: {str(e)}")
            
            # Si falla, intentar usar el tokenizador metacognitivo español
            ruta_tokenizador = "D:/AItrainer/Data/Tokenizers/Spanish/tokenizador_metacognitivo"
            logger.info(f"Intentando cargar tokenizador metacognitivo español desde: {ruta_tokenizador}")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(ruta_tokenizador)
                logger.info("Tokenizador metacognitivo español cargado correctamente")
            except Exception as e:
                logger.error(f"Error cargando tokenizador metacognitivo específico: {str(e)}")
                logger.warning("Usando tokenizador de respaldo GPT2")
                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        # Verificar que el tokenizador tenga token de padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Modelo y tokenizador cargados correctamente")
        return modelo, tokenizer
    
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description='Probar uno o varios modelos de lenguaje entrenados con enfoque metacognitivo directo'
    )
    parser.add_argument('--ruta_modelo', type=str, required=True,
                      help='Ruta al directorio donde está guardado el modelo. Para probar múltiples modelos, separe las rutas con comas (ejemplo: "ruta/modelo1,ruta/modelo2")')
    parser.add_argument('--prompt', type=str, default="Hola, soy un modelo de lenguaje que",
                      help='Texto inicial para la generación')
    parser.add_argument('--max_length', type=int, default=200,
                      help='Longitud máxima del texto generado')
    parser.add_argument('--temperatura', type=float, default=0.8,
                      help='Temperatura para sampling (más alta = más aleatorio)')
    parser.add_argument('--num_secuencias', type=int, default=1,
                      help='Número de secuencias a generar')
    parser.add_argument('--cpu', action='store_true',
                      help='Usar CPU en lugar de GPU')
    parser.add_argument('--log_file', type=str, default=None,
                      help='Ruta donde guardar el archivo de log. Si no se especifica, solo se muestra en consola')
    
    args = parser.parse_args()
    
    # Configurar logging basado en si se especificó un archivo de log
    handlers = [logging.StreamHandler()]
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, mode='w'))
        print(f"El log se guardará en: {os.path.abspath(args.log_file)}")
    
    # Configurar el logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Determinar dispositivo
    if args.cpu or not torch.cuda.is_available():
        device = "cpu"
        logger.info("Usando CPU para inferencia")
    else:
        device = "cuda"
        logger.info(f"Usando GPU {torch.cuda.get_device_name(0)} para inferencia")
    
    # Procesar rutas de modelos (pueden ser múltiples separadas por comas)
    rutas_modelos = [ruta.strip() for ruta in args.ruta_modelo.split(',')]
    logger.info(f"Se probarán {len(rutas_modelos)} modelo(s): {rutas_modelos}")
    
    # Iterar por cada modelo y probarlo
    for i, ruta_modelo in enumerate(rutas_modelos):
        try:
            logger.info(f"\n{'='*50}\nPROBANDO MODELO {i+1}/{len(rutas_modelos)}: {ruta_modelo}\n{'='*50}")
            
            # Cargar modelo y tokenizer
            modelo, tokenizer = cargar_modelo(ruta_modelo, device)
            
            if modelo is None or tokenizer is None:
                logger.error(f"No se pudo cargar el modelo o tokenizer de {ruta_modelo}. Saltando al siguiente.")
                continue
            
            # Probar generación
            try:
                probar_generacion(
                    modelo, 
                    tokenizer, 
                    args.prompt, 
                    max_length=args.max_length,
                    temperatura=args.temperatura,
                    num_secuencias=args.num_secuencias,
                    device=device
                )
            except Exception as e:
                logger.error(f"Error durante la generación de texto: {str(e)}")
                logger.error("Continuando con el siguiente modelo...")
                import traceback
                logger.error(traceback.format_exc())
            
            # Liberar memoria GPU
            if device == "cuda":
                try:
                    del modelo
                    torch.cuda.empty_cache()
                    logger.info("Memoria GPU liberada")
                except Exception as e:
                    logger.error(f"Error al liberar memoria GPU: {str(e)}")
        except Exception as e:
            logger.error(f"Error inesperado al probar el modelo {ruta_modelo}: {str(e)}")
            logger.error("Continuando con el siguiente modelo...")
            import traceback
            logger.error(traceback.format_exc())
        
    logger.info("\nPrueba de todos los modelos completada")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Proceso interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
