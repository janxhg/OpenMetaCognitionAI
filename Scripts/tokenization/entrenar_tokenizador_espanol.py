#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entrenamiento de Tokenizador Especializado para Español

Este script entrena un tokenizador especializado en español utilizando 
los datos disponibles en el sistema. El tokenizador resultante mejora 
significativamente la representación de texto en español, reduciendo 
la fragmentación de palabras y optimizando el uso de tokens.

El enfoque metacognitivo del entrenamiento permite al tokenizador adaptarse
específicamente a las características morfológicas y sintácticas del español.
"""

import os
import glob
import time
import json
import shutil
import logging
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast

# Añadir rutas para importar módulos del directorio core si es necesario
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Importar componentes si son necesarios
try:
    from tokenization.tokenizador_metacognitivo import MetacognitivoTokenizer
except ImportError as e:
    print(f"No se pudo importar el tokenizador metacognitivo: {e}")
    print("No es crítico para el entrenamiento inicial del tokenizador")

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tokenizador_espanol.log')
    ]
)
logger = logging.getLogger(__name__)


def encontrar_archivos_texto(directorios: List[str], 
                             extensiones: List[str] = ['.txt'], 
                             max_archivos: int = 1000) -> List[str]:
    """
    Encuentra todos los archivos de texto dentro de los directorios especificados.
    
    Args:
        directorios: Lista de directorios donde buscar
        extensiones: Extensiones de archivo a considerar
        max_archivos: Número máximo de archivos a utilizar
        
    Returns:
        Lista de rutas a archivos encontrados
    """
    archivos = []
    total_encontrados = 0
    
    for directorio in directorios:
        if not os.path.exists(directorio):
            logger.warning(f"El directorio {directorio} no existe.")
            continue
            
        logger.info(f"Buscando archivos de texto en {directorio}")
        for extension in extensiones:
            patron = os.path.join(directorio, f"**/*{extension}")
            encontrados = glob.glob(patron, recursive=True)
            logger.info(f"  Encontrados {len(encontrados)} archivos con extensión {extension}")
            archivos.extend(encontrados)
            total_encontrados += len(encontrados)
            
            if total_encontrados >= max_archivos:
                logger.warning(f"Se alcanzó el límite de {max_archivos} archivos.")
                return archivos[:max_archivos]
    
    if not archivos:
        logger.error("No se encontraron archivos de texto para entrenar el tokenizador.")
    else:
        logger.info(f"Total de archivos encontrados: {len(archivos)}")
        
    return archivos


def entrenar_tokenizador(archivos: List[str], 
                         dir_salida: str, 
                         vocab_size: int = 30000,
                         min_frequency: int = 2,
                         batch_files: int = 50,
                         mem_efficient: bool = False) -> None:
    """
    Entrena un tokenizador ByteLevelBPE especializado en español.
    
    Args:
        archivos: Lista de archivos para entrenamiento
        dir_salida: Directorio donde guardar el tokenizador
        vocab_size: Tamaño del vocabulario
        min_frequency: Frecuencia mínima para incluir un token
    """
    if not archivos:
        logger.error("No hay archivos para entrenar el tokenizador.")
        return
    
    # Crear directorio de salida si no existe
    os.makedirs(dir_salida, exist_ok=True)
    
    # Verificar acceso a archivos
    archivos_accesibles = []
    for archivo in archivos:
        try:
            with open(archivo, 'r', encoding='utf-8') as f:
                # Solo verificar que se puede leer
                f.read(1024)  # Leer los primeros 1024 bytes
            archivos_accesibles.append(archivo)
        except Exception as e:
            logger.warning(f"No se puede acceder al archivo {archivo}: {str(e)}")
    
    if not archivos_accesibles:
        logger.error("Ninguno de los archivos es accesible para entrenamiento.")
        return
        
    logger.info(f"Iniciando entrenamiento con {len(archivos_accesibles)} archivos")
    logger.info(f"Parámetros: vocab_size={vocab_size}, min_frequency={min_frequency}")
    
    # Tokens especiales para compatibilidad con modelos tipo GPT-2
    special_tokens = [
        "<s>",      # Inicio de secuencia
        "</s>",     # Fin de secuencia
        "<pad>",    # Padding
        "<unk>",    # Token desconocido
        "<mask>",   # Token de máscara para modelos tipo BERT
    ]
    
    # Inicializar tokenizador
    tokenizer = ByteLevelBPETokenizer()
    
    # Entrenar el tokenizador
    try:
        inicio = time.time()
        
        if mem_efficient:
            logger.info(f"Usando modo de memoria eficiente con {batch_files} archivos por lote")
            entrenar_por_lotes(tokenizer, archivos_accesibles, vocab_size, min_frequency, 
                              special_tokens, batch_files, dir_salida)
        else:
            logger.info("Entrenando tokenizador con todos los archivos a la vez")
            tokenizer.train(
                files=archivos_accesibles,
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=special_tokens
            )
        
        duracion = time.time() - inicio
        logger.info(f"Entrenamiento completado en {duracion:.2f} segundos")
        
        # Guardar el tokenizador base
        ruta_base = os.path.join(dir_salida, "tokenizador_base")
        os.makedirs(ruta_base, exist_ok=True)
        tokenizer.save_model(ruta_base)
        logger.info(f"Tokenizador base guardado en {ruta_base}")
        
        # Convertir a formato compatible con identidad metacognitiva
        ruta_metacog = os.path.join(dir_salida, "tokenizador_metacognitivo")
        os.makedirs(ruta_metacog, exist_ok=True)
        
        logger.info("Convirtiendo tokenizador al formato con identidad metacognitiva...")
        
        try:
            # Intentar cargar el tokenizador
            with open(os.path.join(ruta_base, "vocab.json"), 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            
            # Guardar vocab.json
            with open(os.path.join(ruta_metacog, "vocab.json"), 'w', encoding='utf-8') as f:
                json.dump(vocab, f, ensure_ascii=False)
            
            # Copiar merges.txt
            shutil.copy(
                os.path.join(ruta_base, "merges.txt"),
                os.path.join(ruta_metacog, "merges.txt")
            )
            
            # Crear configuración personalizada
            tokenizer_config = {
                "model_type": "metacognitivo_espanol",
                "tokenizer_class": "MetacognitivoTokenizer",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "special_tokens_map": {
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "pad_token": "<pad>",
                    "unk_token": "<unk>",
                    "additional_special_tokens": [
                        "<metacog>",
                        "</metacog>",
                        "<reflect>",
                        "<introspect>"
                    ]
                },
                "clean_up_tokenization_spaces": True
            }
            
            with open(os.path.join(ruta_metacog, "tokenizer_config.json"), 'w', encoding='utf-8') as f:
                json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
            
            # Crear special_tokens_map.json
            special_tokens_map = {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "additional_special_tokens": [
                    "<metacog>", 
                    "</metacog>", 
                    "<reflect>",
                    "<introspect>"
                ]
            }
            
            with open(os.path.join(ruta_metacog, "special_tokens_map.json"), 'w', encoding='utf-8') as f:
                json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)
            
            # Crear added_tokens.json (vacío por ahora)
            with open(os.path.join(ruta_metacog, "added_tokens.json"), 'w', encoding='utf-8') as f:
                json.dump({}, f)
            
            logger.info(f"Tokenizador con identidad metacognitiva guardado en {ruta_metacog}")
            
            # Para compatibilidad, también crear versión GPT-2 estándar
            ruta_gpt2 = os.path.join(dir_salida, "tokenizador_gpt2")
            os.makedirs(ruta_gpt2, exist_ok=True)
            
            # Copiar archivos básicos
            shutil.copy(os.path.join(ruta_base, "vocab.json"), os.path.join(ruta_gpt2, "vocab.json"))
            shutil.copy(os.path.join(ruta_base, "merges.txt"), os.path.join(ruta_gpt2, "merges.txt"))
            
            # Configuración GPT-2 estándar
            config_gpt2 = {
                "model_type": "gpt2",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "unk_token": "<unk>"
            }
            
            with open(os.path.join(ruta_gpt2, "config.json"), 'w', encoding='utf-8') as f:
                json.dump(config_gpt2, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Tokenizador formato GPT-2 guardado en {ruta_gpt2}")
            logger.info(f"Tokenizador compatible con GPT-2 guardado en {ruta_gpt2}")
        except Exception as e:
            logger.error(f"Error al convertir al formato metacognitivo: {str(e)}")
            raise
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento del tokenizador: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Intentar recuperación usando el script arreglar_tokenizador.py
        logger.info("Intentando recuperación mediante el script arreglar_tokenizador.py")
        try:
            from arreglar_tokenizador import arreglar_tokenizador as fix_tokenizer
            exito = fix_tokenizer(
                dir_base=ruta_base,
                dir_salida=os.path.join(dir_salida, "tokenizador_recuperado"),
                nombre_modelo="metacognitivo_espanol"
            )
            if exito:
                logger.info("Recuperación exitosa del tokenizador")
            else:
                logger.error("No se pudo recuperar el tokenizador")
        except Exception as e2:
            logger.error(f"Error en la recuperación: {str(e2)}")


def entrenar_por_lotes(tokenizer, archivos, vocab_size, min_frequency, special_tokens, batch_size, dir_salida):
    """Entrena el tokenizador procesando archivos en lotes para evitar problemas de memoria.
    
    Args:
        tokenizer: Tokenizador a entrenar
        archivos: Lista de rutas a archivos para entrenamiento
        vocab_size: Tamaño del vocabulario
        min_frequency: Frecuencia mínima para incluir token
        special_tokens: Lista de tokens especiales
        batch_size: Número de archivos a procesar en cada lote
        dir_salida: Directorio donde guardar checkpoints
    """
    import random
    import gc
    from itertools import islice
    
    # Mezclar archivos para mejor distribución de datos
    random.shuffle(archivos)
    
    total_batches = (len(archivos) + batch_size - 1) // batch_size
    logger.info(f"Procesando {len(archivos)} archivos en {total_batches} lotes")
    
    # Inicialización del tokenizador en el primer lote
    first_batch = True
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(archivos))
        batch_files = archivos[start_idx:end_idx]
        
        logger.info(f"Procesando lote {i+1}/{total_batches} ({len(batch_files)} archivos)")
        
        # Generar un iterador de texto desde los archivos del lote
        text_iterator = yield_text_from_files(batch_files)
        
        if first_batch:
            # Primer lote: entrenar desde cero
            tokenizer.train_from_iterator(
                text_iterator,
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=special_tokens
            )
            first_batch = False
        else:
            # Lotes siguientes: continuar el entrenamiento
            tokenizer.train_from_iterator(
                text_iterator,
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=special_tokens
            )
        
        # Liberar memoria
        gc.collect()
        
        # Guardar checkpoint cada 5 lotes o en el último lote
        if (i+1) % 5 == 0 or (i+1) == total_batches:
            checkpoint_path = os.path.join(dir_salida, f"checkpoint_batch_{i+1}")
            os.makedirs(checkpoint_path, exist_ok=True)
            tokenizer.save_model(checkpoint_path)
            logger.info(f"Checkpoint guardado en {checkpoint_path}")


def yield_text_from_files(files, chunk_size=1024*1024):
    """Generador que lee archivos en bloques para evitar cargarlos todos en memoria.
    
    Args:
        files: Lista de archivos a leer
        chunk_size: Tamaño del bloque de lectura en bytes
        
    Yields:
        Líneas de texto de los archivos
    """
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                # Leer el archivo por bloques
                for chunk in iter(lambda: f.read(chunk_size), ''):
                    # Dividir por líneas, evitando cortar palabras entre bloques
                    lines = chunk.splitlines()
                    for line in lines:
                        if line.strip():  # Omitir líneas vacías
                            yield line
        except Exception as e:
            logger.warning(f"Error leyendo {file}: {str(e)}")


def evaluar_tokenizador(ruta_tokenizador: str, textos_prueba: List[str]) -> Dict[str, Any]:
    """
    Evalúa el tokenizador entrenado con textos de prueba.
    
    Args:
        ruta_tokenizador: Ruta al tokenizador en formato GPT-2
        textos_prueba: Lista de textos para evaluar
        
    Returns:
        Diccionario con métricas de evaluación
    """
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(ruta_tokenizador)
        logger.info(f"Tokenizador cargado desde {ruta_tokenizador}")
        
        # Evaluar con tokenizador estándar de GPT-2 para comparación
        tokenizer_base = GPT2TokenizerFast.from_pretrained("gpt2")
        logger.info("Tokenizador GPT-2 base cargado para comparación")
        
        resultados = {
            "textos_evaluados": len(textos_prueba),
            "promedio_tokens_base": 0,
            "promedio_tokens_espanol": 0,
            "reduccion_porcentaje": 0,
            "detalles_por_texto": []
        }
        
        total_tokens_base = 0
        total_tokens_espanol = 0
        
        for i, texto in enumerate(textos_prueba):
            # Tokenizar con GPT-2 base
            tokens_base = tokenizer_base.encode(texto)
            n_tokens_base = len(tokens_base)
            total_tokens_base += n_tokens_base
            
            # Tokenizar con nuestro tokenizador español
            tokens_espanol = tokenizer.encode(texto)
            n_tokens_espanol = len(tokens_espanol)
            total_tokens_espanol += n_tokens_espanol
            
            # Calcular reducción para este texto
            reduccion = (1 - n_tokens_espanol / n_tokens_base) * 100 if n_tokens_base > 0 else 0
            
            resultados["detalles_por_texto"].append({
                "texto_id": i,
                "texto": texto[:50] + "..." if len(texto) > 50 else texto,
                "tokens_base": n_tokens_base,
                "tokens_espanol": n_tokens_espanol,
                "reduccion_porcentaje": reduccion
            })
            
            if i < 3:  # Mostrar ejemplos de tokenización para los primeros textos
                logger.info(f"\nEjemplo {i+1}:")
                logger.info(f"Texto: {texto[:100]}...")
                logger.info(f"Tokens GPT-2 base ({n_tokens_base}): {tokenizer_base.convert_ids_to_tokens(tokens_base)[:20]}...")
                logger.info(f"Tokens Español ({n_tokens_espanol}): {tokenizer.convert_ids_to_tokens(tokens_espanol)[:20]}...")
                logger.info(f"Reducción: {reduccion:.2f}%")
        
        # Calcular promedios
        if textos_prueba:
            resultados["promedio_tokens_base"] = total_tokens_base / len(textos_prueba)
            resultados["promedio_tokens_espanol"] = total_tokens_espanol / len(textos_prueba)
            resultados["reduccion_porcentaje"] = (1 - total_tokens_espanol / total_tokens_base) * 100 if total_tokens_base > 0 else 0
        
        logger.info("\n--- Resultados de Evaluación ---")
        logger.info(f"Textos evaluados: {resultados['textos_evaluados']}")
        logger.info(f"Promedio tokens GPT-2 base: {resultados['promedio_tokens_base']:.2f}")
        logger.info(f"Promedio tokens español: {resultados['promedio_tokens_espanol']:.2f}")
        logger.info(f"Reducción promedio: {resultados['reduccion_porcentaje']:.2f}%")
        
        return resultados
        
    except Exception as e:
        logger.error(f"Error durante la evaluación del tokenizador: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def obtener_texto_prueba(archivos: List[str], 
                         max_textos: int = 5, 
                         max_longitud: int = 300) -> List[str]:
    """
    Obtiene textos de prueba de los archivos disponibles.
    
    Args:
        archivos: Lista de archivos de donde extraer textos
        max_textos: Número máximo de textos a extraer
        max_longitud: Longitud máxima de cada texto
        
    Returns:
        Lista de textos de prueba
    """
    textos = []
    
    if not archivos:
        return textos
    
    # Mezclar y limitar archivos
    import random
    random.shuffle(archivos)
    archivos_muestra = archivos[:min(10, len(archivos))]
    
    for archivo in archivos_muestra:
        try:
            with open(archivo, 'r', encoding='utf-8') as f:
                contenido = f.read()
                
                # Dividir en párrafos o líneas
                partes = contenido.split('\n\n')
                if len(partes) < 2:
                    partes = contenido.split('\n')
                
                # Tomar algunas partes aleatorias
                if partes:
                    random.shuffle(partes)
                    for parte in partes[:2]:  # Tomar hasta 2 partes por archivo
                        if len(parte) > 20:  # Solo si tiene contenido significativo
                            textos.append(parte[:max_longitud])
                            if len(textos) >= max_textos:
                                return textos
                                
        except Exception as e:
            logger.warning(f"Error al leer texto de prueba de {archivo}: {str(e)}")
    
    return textos


def guardar_metadatos(dir_salida: str, 
                     archivos_usados: List[str], 
                     vocab_size: int,
                     resultados_evaluacion: Dict[str, Any]) -> None:
    """
    Guarda metadatos del proceso de entrenamiento para referencia futura.
    
    Args:
        dir_salida: Directorio donde guardar los metadatos
        archivos_usados: Lista de archivos usados en el entrenamiento
        vocab_size: Tamaño del vocabulario
        resultados_evaluacion: Resultados de la evaluación
    """
    import json
    from datetime import datetime
    
    metadatos = {
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "archivos_usados": len(archivos_usados),
        "muestra_archivos": archivos_usados[:10] if len(archivos_usados) > 10 else archivos_usados,
        "parametros": {
            "vocab_size": vocab_size,
            "min_frequency": 2,
        },
        "evaluacion": resultados_evaluacion
    }
    
    ruta_metadatos = os.path.join(dir_salida, "metadatos_tokenizador.json")
    try:
        with open(ruta_metadatos, 'w', encoding='utf-8') as f:
            json.dump(metadatos, f, ensure_ascii=False, indent=2)
        logger.info(f"Metadatos guardados en {ruta_metadatos}")
    except Exception as e:
        logger.error(f"Error al guardar metadatos: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='Entrenamiento de tokenizador especializado para español'
    )
    
    # Argumentos para directorios de datos
    parser.add_argument('--directorios', type=str, nargs='+', 
                       help='Directorios donde buscar archivos de texto en español')
    parser.add_argument('--dir_salida', type=str, default='D:\\AItrainer\\Data\\Tokenizers\\Spanish',
                       help='Directorio donde guardar el tokenizador entrenado')
    
    # Parámetros de configuración
    parser.add_argument('--vocab_size', type=int, default=30000,
                       help='Tamaño del vocabulario del tokenizador')
    parser.add_argument('--min_frequency', type=int, default=2,
                       help='Frecuencia mínima para incluir un token')
    parser.add_argument('--max_archivos', type=int, default=100,
                       help='Número máximo de archivos a usar para entrenamiento')
    parser.add_argument('--batch_files', type=int, default=50,
                       help='Número de archivos a procesar en cada lote')
    parser.add_argument('--mem_efficient', action='store_true',
                       help='Usar modo de memoria eficiente para datasets grandes')
    
    # Opciones adicionales
    parser.add_argument('--solo_evaluar', action='store_true',
                       help='Solo evaluar un tokenizador existente sin entrenamiento')
    parser.add_argument('--ruta_tokenizador', type=str,
                       help='Ruta al tokenizador para evaluación (solo con --solo_evaluar)')
    
    args = parser.parse_args()
    
    # Directorio de salida por defecto si no se especifica
    if not args.dir_salida:
        args.dir_salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "tokenizers", "spanish")
    
    # Establecer directorios por defecto si no se especifican
    if not args.directorios:
        args.directorios = [
            "D:\\AItrainer\\Data\\DataBase\\español\\Spanish",
            "D:\\AItrainer\\Data\\DataBase\\español\\ChunkedFiles",
        ]
        logger.info(f"Usando directorios por defecto: {args.directorios}")
    
    # Crear directorio de salida
    os.makedirs(args.dir_salida, exist_ok=True)
    
    # Flujo principal
    if args.solo_evaluar and args.ruta_tokenizador:
        # Solo evaluación
        archivos = encontrar_archivos_texto(args.directorios, max_archivos=args.max_archivos)
        textos_prueba = obtener_texto_prueba(archivos)
        evaluar_tokenizador(args.ruta_tokenizador, textos_prueba)
    else:
        # Entrenamiento completo
        logger.info("Iniciando proceso de entrenamiento de tokenizador español")
        
        # Encontrar archivos para entrenamiento
        archivos = encontrar_archivos_texto(args.directorios, max_archivos=args.max_archivos)
        if not archivos:
            logger.error("No se encontraron archivos para entrenar el tokenizador.")
            return
        
        # Entrenar tokenizador
        entrenar_tokenizador(
            archivos=archivos,
            dir_salida=args.dir_salida,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            batch_files=args.batch_files,
            mem_efficient=args.mem_efficient
        )
        
        # Evaluar el tokenizador entrenado (usar preferentemente el metacognitivo)
        ruta_tokenizador = os.path.join(args.dir_salida, "tokenizador_metacognitivo")
        if not os.path.exists(ruta_tokenizador):
            # Si no existe el metacognitivo, intentar con el GPT-2 o el base
            ruta_tokenizador = os.path.join(args.dir_salida, "tokenizador_gpt2")
            if not os.path.exists(ruta_tokenizador):
                ruta_tokenizador = os.path.join(args.dir_salida, "tokenizador_base")
                
        textos_prueba = obtener_texto_prueba(archivos)
        resultados = evaluar_tokenizador(ruta_tokenizador, textos_prueba)
        
        # Guardar metadatos
        guardar_metadatos(
            dir_salida=args.dir_salida,
            archivos_usados=archivos,
            vocab_size=args.vocab_size,
            resultados_evaluacion=resultados
        )
        
        logger.info(f"Proceso completado. Tokenizador guardado en {args.dir_salida}")


if __name__ == "__main__":
    try:
        start_time = time.time()
        main()
        duration = time.time() - start_time
        logger.info(f"Proceso completo en {duration:.2f} segundos")
    except KeyboardInterrupt:
        logger.info("Proceso interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error en el proceso: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
