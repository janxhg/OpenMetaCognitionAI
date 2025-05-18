#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entrenar con Metacognitivo Directo

Este script implementa un sistema alternativo de entrenamiento que reemplaza
el concepto tradicional de épocas por un enfoque metacognitivo directo,
donde el modelo controla activamente su propio proceso de aprendizaje.
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configurar correctamente las rutas para encontrar todos los módulos
# Primero obtenemos la ruta base del proyecto
script_dir = os.path.dirname(os.path.abspath(__file__))
proyect_dir = script_dir  # La carpeta principal es donde está este script
scripts_dir = os.path.join(proyect_dir, 'Scripts')  # Nueva carpeta Scripts

# Añadir todas las rutas importantes al path de Python
sys.path.insert(0, proyect_dir)  # Añadir la carpeta raíz
sys.path.insert(0, scripts_dir)  # Añadir la carpeta Scripts
sys.path.insert(0, os.path.join(scripts_dir, 'core'))  # Añadir la carpeta core
sys.path.insert(0, os.path.join(scripts_dir, 'training'))  # Añadir la carpeta training
sys.path.insert(0, os.path.join(scripts_dir, 'utils'))  # Añadir la carpeta utils
sys.path.insert(0, os.path.join(scripts_dir, 'tokenization'))  # Añadir la carpeta tokenization

# Imprimir rutas para debug
print("Rutas de búsqueda de Python:")
for ruta in sys.path[:5]:  # Mostrar las primeras 5 rutas
    print(f"  - {ruta}")


# Configuración de logging primero para poder usarlo de inmediato
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('entrenamiento_directo.log', 'a')
    ]
)
logger = logging.getLogger(__name__)

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Importar el sistema metacognitivo integrado y sus componentes
# No necesitamos ajustar el path ya que estamos en la raíz del proyecto

# Importar desde core, training y utils
try:
    # Primero importar módulos utils que puedan ser requeridos por otros módulos
    from Scripts.utils.crear_modelo_desde_cero import crear_modelo_y_tokenizador_nuevos, extraer_textos_de_materiales
    
    # Luego importar módulos core
    from Scripts.core.sistema_metacognitivo_integrado import SistemaMetacognitivoIntegrado
    from Scripts.core.auto_observacion import AutoObservacion
    from Scripts.core.reflexion_metacognitiva import ReflexionMetacognitiva
    from Scripts.core.auto_modificacion import AutoModificacion
    from Scripts.core.cerebro_autonomo import CerebroAutonomo
    
    # Importar desde el directorio training
    from Scripts.training.entrenamiento_metacognitivo_directo import (
        entrenar_modelo_metacognitivo_directo,
        EntrenamientoDirectoCallback
    )
    
    logger.info("Módulos metacognitivos importados correctamente")
except ImportError as e:
    logger.error(f"Error al importar módulos metacognitivos: {str(e)}")
    logger.error("Asegúrate de que todos los componentes necesarios están presentes en las carpetas 'core' y 'training'")
    sys.exit(1)

# No usaremos mejoras_metacognitivas en este enfoque
AUTO_ADAPTACION_DISPONIBLE = False


def obtener_textos_entrenamiento(directorio: str, excluir: Optional[List[str]] = None) -> List[str]:
    """
    Obtiene todos los textos disponibles para entrenamiento.
    Busca recursivamente en el directorio y sus subdirectorios.
    
    Args:
        directorio: Directorio donde se encuentran los textos
        excluir: Lista de archivos o patrones a excluir
        
    Returns:
        Lista de textos para entrenamiento
    """
    textos = []
    archivos_procesados = 0
    archivos_excluidos = 0
    
    # Validar que el directorio existe
    if not os.path.exists(directorio):
        logger.error(f"El directorio {directorio} no existe")
        return []
    
    logger.info(f"Buscando archivos .txt en {directorio} y subdirectorios")
    
    # Normalizar los patrones de exclusión
    patrones_exclusion = []
    if excluir:
        patrones_exclusion = [patron.lower() for patron in excluir if patron.strip()]
    
    # Función recursiva para buscar en subdirectorios
    def procesar_directorio(ruta):
        nonlocal textos, archivos_procesados, archivos_excluidos
        
        # Verificar si es un directorio válido
        if not os.path.isdir(ruta):
            return
            
        try:
            # Listar contenidos del directorio
            contenidos = os.listdir(ruta)
            logger.info(f"Explorando {ruta}: encontrados {len(contenidos)} elementos")
            
            # Primero procesar archivos .txt
            for nombre in contenidos:
                ruta_completa = os.path.join(ruta, nombre)
                
                # Saltar subdirectorios para procesarlos después
                if os.path.isdir(ruta_completa):
                    continue
                    
                # Sólo procesar archivos .txt
                if not nombre.lower().endswith('.txt'):
                    continue
                
                # Verificar si debe excluirse
                excluir_archivo = False
                nombre_lower = nombre.lower()
                for patron in patrones_exclusion:
                    if patron in nombre_lower:
                        excluir_archivo = True
                        archivos_excluidos += 1
                        break
                
                if excluir_archivo:
                    continue
                
                # Leer el contenido del archivo
                try:
                    with open(ruta_completa, 'r', encoding='utf-8') as f:
                        contenido = f.read().strip()
                        if contenido:  # Asegurar que no esté vacío
                            textos.append(contenido)
                            archivos_procesados += 1
                            if archivos_procesados % 10 == 0:
                                logger.info(f"Procesados {archivos_procesados} archivos hasta ahora")
                except Exception as e:
                    logger.warning(f"Error al leer {ruta_completa}: {str(e)}")
            
            # Ahora procesar subdirectorios recursivamente
            for nombre in contenidos:
                ruta_completa = os.path.join(ruta, nombre)
                if os.path.isdir(ruta_completa):
                    procesar_directorio(ruta_completa)
                    
        except Exception as e:
            logger.warning(f"Error al explorar directorio {ruta}: {str(e)}")
    
    # Iniciar procesamiento recursivo
    procesar_directorio(directorio)
    
    logger.info(f"Procesados {archivos_procesados} archivos, excluidos {archivos_excluidos}")
    return textos


def limpiar_modelos_anteriores(directorio: str, max_modelos: int = 5, solo_mejor: bool = False):
    """
    Limpia modelos anteriores guardados en el directorio.
    
    Args:
        directorio: Directorio donde se encuentran los modelos
        max_modelos: Número máximo de modelos a mantener
        solo_mejor: Si es True, solo mantiene el mejor modelo
    """
    import shutil
    from os.path import isdir, join
    
    # Obtener todos los subdirectorios (modelos guardados)
    subdirs = [d for d in os.listdir(directorio) if isdir(join(directorio, d))]
    if not subdirs:
        return
    
    # Filtrar solo los directorios de modelos (contienen config.json)
    modelo_dirs = []
    for d in subdirs:
        modelo_path = join(directorio, d)
        if os.path.exists(join(modelo_path, 'config.json')):
            # Obtener timestamp del nombre o de la fecha de modificación
            try:
                timestamp = int(d.split('_')[-1])
            except:
                timestamp = os.path.getmtime(modelo_path)
            modelo_dirs.append((modelo_path, timestamp))
    
    # Si no hay suficientes modelos para alcanzar el límite, no hacer nada
    if len(modelo_dirs) <= max_modelos and not solo_mejor:
        return
    
    # Ordenar por timestamp (más reciente primero)
    modelo_dirs.sort(key=lambda x: x[1], reverse=True)
    
    # Si solo_mejor es True, mantener solo el último modelo
    if solo_mejor and modelo_dirs:
        a_mantener = [modelo_dirs[0][0]]
        a_eliminar = [path for path, _ in modelo_dirs[1:]]
    else:
        # Mantener los max_modelos más recientes
        a_mantener = [path for path, _ in modelo_dirs[:max_modelos]]
        a_eliminar = [path for path, _ in modelo_dirs[max_modelos:]]
    
    # Eliminar los modelos sobrantes
    for path in a_eliminar:
        try:
            logger.info(f"Eliminando modelo antiguo: {path}")
            shutil.rmtree(path)
        except Exception as e:
            logger.warning(f"Error al eliminar {path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='Entrenamiento metacognitivo directo (sin épocas tradicionales)'
    )
    
    # Argumentos para el modelo y datos
    parser.add_argument('--modelo_base', type=str, default='nuevo',
                      help='Nombre del modelo base (o "nuevo" para crear uno nuevo)')
    parser.add_argument('--dir_trabajo', type=str, required=True,
                      help='Directorio de trabajo donde se guarda el modelo y resultados')
    parser.add_argument('--dataset_externo', type=str, default=None,
                      help='Ruta a un directorio con archivos .txt para entrenamiento')
    parser.add_argument('--nombre_modelo', type=str, default='modelo_metacognitivo_directo',
                      help='Nombre para guardar el modelo entrenado')
    
    # Parámetros del entrenamiento directo
    parser.add_argument('--pasos', type=int, default=100,
                      help='Número de pasos de entrenamiento (reemplaza épocas)')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Tamaño de batch para entrenamiento')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Tasa de aprendizaje para optimizador')
    parser.add_argument('--max_length', type=int, default=128,
                      help='Longitud máxima de secuencia para tokenización')
    
    # Parámetros para limpieza de modelos
    parser.add_argument('--max_modelos_guardados', type=int, default=0,
                      help='Máximo número de modelos a mantener guardados (0 = sin límite)')
    parser.add_argument('--guardar_solo_mejor', action='store_true',
                      help='Guardar solo el mejor modelo (más reciente)')
    parser.add_argument('--limpiar_modelos_anteriores', action='store_true',
                      help='Limpiar modelos anteriores antes de entrenar')
    parser.add_argument('--checkpoint_cada', type=int, default=0,
                      help='Guardar checkpoints cada N pasos (0 para desactivar, recomendado: 1000)')
    
    # Parámetros específicos del entrenamiento metacognitivo
    parser.add_argument('--ciclos', type=int, default=5,
                      help='Número de ciclos metacognitivos a ejecutar')
    parser.add_argument('--nivel_inteligencia', type=int, default=5,
                      help='Nivel de inteligencia metacognitiva (1-10)')
    parser.add_argument('--fp16', action='store_true',
                      help='Usar entrenamiento en precisión mixta (FP16) para mayor eficiencia')
    parser.add_argument('--usar_cerebro_autonomo', action='store_true',
                      help='Habilitar enfoque basado en CerebroAutonomo para el entrenamiento (recomendado)')
    
    # Parámetros para configurar el tamaño del modelo (cuando modelo_base='nuevo')
    parser.add_argument('--n_embd', type=int, default=256,
                      help='Dimensiones de embeddings (default: 256 para ~2M params, 768 para ~100M params)')
    parser.add_argument('--n_layer', type=int, default=4,
                      help='Número de capas del transformador (default: 4 para ~2M params, 16 para ~100M params)')
    parser.add_argument('--n_head', type=int, default=4,
                      help='Número de cabezas de atención (default: 4 para ~2M params, 12 para ~100M params)')
    parser.add_argument('--n_ctx', type=int, default=128,
                      help='Tamaño del contexto en tokens (default: 128)'
                     )
    
    # Opciones adicionales
    parser.add_argument('--archivos_excluidos', type=str, default='',
                      help='Patrones de archivos a excluir del entrenamiento (separados por comas)')
    
    args = parser.parse_args()
    
    # Configurar directorios
    os.makedirs(args.dir_trabajo, exist_ok=True)
    dir_modelos = os.path.join(args.dir_trabajo, 'modelos')
    dir_logs = os.path.join(args.dir_trabajo, 'logs')
    os.makedirs(dir_modelos, exist_ok=True)
    os.makedirs(dir_logs, exist_ok=True)
    
    # Limpiar modelos anteriores si está habilitado
    if args.limpiar_modelos_anteriores and args.max_modelos_guardados > 0:
        limpiar_modelos_anteriores(
            dir_modelos, 
            args.max_modelos_guardados, 
            args.guardar_solo_mejor
        )
    
    # Inicializar sistema metacognitivo
    logger.info("Inicializando sistema metacognitivo integrado")
    patrones_excluidos = [p.strip() for p in args.archivos_excluidos.split(',')] if args.archivos_excluidos else []
    
    sistema = SistemaMetacognitivoIntegrado(
        args.modelo_base,
        dir_trabajo=args.dir_trabajo,
        nivel_inteligencia=args.nivel_inteligencia
    )
    
    # Obtener textos para entrenamiento
    textos = []
    if args.dataset_externo and os.path.exists(args.dataset_externo):
        logger.info(f"Cargando textos desde {args.dataset_externo}")
        textos = obtener_textos_entrenamiento(args.dataset_externo, patrones_excluidos)
    
    if not textos:
        logger.error("No se encontraron textos para entrenamiento")
        return
    
    logger.info(f"Iniciando entrenamiento metacognitivo directo con {len(textos)} textos")
    
    # Crear callback para integración con el sistema existente
    callback = EntrenamientoDirectoCallback(
        auto_observacion=sistema.introspector if hasattr(sistema, 'introspector') else None,
        reflexion=sistema.reflexion if hasattr(sistema, 'reflexion') else None,
        auto_modificacion=sistema.modificador if hasattr(sistema, 'modificador') else None
    )
    
    # Utilizamos el sistema metacognitivo basado en CerebroAutonomo sin auto-adaptación
    logger.info("Utilizando enfoque basado en CerebroAutonomo")
    adaptador_metacognitivo = None
    
    # Verificamos que el sistema tiene los componentes necesarios
    if hasattr(sistema, 'cerebro') and sistema.cerebro is not None:
        logger.info(f"CerebroAutonomo inicializado correctamente")
    else:
        logger.warning("CerebroAutonomo no inicializado. El sistema puede no funcionar correctamente.")
    
    # Verificar si el modelo está inicializado correctamente
    if sistema.modelo is None:
        logger.warning("El modelo no se ha inicializado correctamente. Creando un modelo totalmente nuevo...")
        
        try:
            # Crear un modelo completamente nuevo con una arquitectura personalizada
            from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
            
            # Configuración personalizable para un modelo nuevo
            n_embd = args.n_embd
            n_layer = args.n_layer
            n_head = args.n_head
            n_ctx = args.n_ctx
            
            # Calcular número aproximado de parámetros
            params_embedding = n_embd * 50257  # vocab_size
            params_per_layer = 12 * n_embd * n_embd  # Aproximación para cada capa transformer
            total_params = params_embedding + (params_per_layer * n_layer)
            total_params_m = total_params / 1000000  # En millones
            
            logger.info(f"Creando modelo con aproximadamente {total_params_m:.2f}M parámetros")
            logger.info(f"Configuración: {n_layer} capas, {n_head} cabezas, {n_embd} dim embeds, {n_ctx} contexto")
            
            configuracion = GPT2Config(
                vocab_size=50257,  # Tamaño de vocabulario estándar
                n_positions=n_ctx,  # Tamaño del contexto
                n_ctx=n_ctx,        # Igual que n_positions
                n_embd=n_embd,      # Dimensiones de embedding 
                n_layer=n_layer,    # Número de capas
                n_head=n_head,      # Cabezas de atención
                resid_pdrop=0.1,    # Dropout para regularización
                embd_pdrop=0.1,     # Dropout para embeddings
                attn_pdrop=0.1,     # Dropout para atención
                layer_norm_epsilon=1e-5,
                initializer_range=0.02,
                bos_token_id=50256,
                eos_token_id=50256
            )
            
            # Crear modelo desde la configuración (sin pesos pre-entrenados)
            logger.info("Inicializando un modelo nuevo desde cero con pesos aleatorios")
            modelo_nuevo = GPT2LMHeadModel(configuracion)
            
            # Usar tokenizador estándar (esto solo define cómo tokenizar, no tiene pesos)
            tokenizador_nuevo = GPT2Tokenizer.from_pretrained("gpt2")
            
            # Establecer en el sistema
            sistema.modelo = modelo_nuevo
            sistema.tokenizer = tokenizador_nuevo
            
            logger.info("Modelo completamente nuevo creado con éxito (arquitectura minimalista)")
        except Exception as e:
            logger.error(f"Error al crear modelo nuevo: {str(e)}")
            return
    
    # Verificar que el tokenizer también está inicializado
    if sistema.tokenizer is None:
        logger.error("No se pudo inicializar el tokenizer")
        return
    
    # Ejecutar entrenamiento directo sin épocas
    try:
        logger.info(f"Usando modelo: {type(sistema.modelo).__name__}")
        resultados = entrenar_modelo_metacognitivo_directo(
            modelo=sistema.modelo,
            tokenizer=sistema.tokenizer,
            textos=textos,
            dir_salida=dir_modelos,
            auto_observacion=sistema.introspector if hasattr(sistema, 'introspector') else None,
            reflexion=sistema.reflexion if hasattr(sistema, 'reflexion') else None,
            auto_modificacion=sistema.modificador if hasattr(sistema, 'modificador') else None,
            pasos_max=args.pasos,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            dispositivo="cuda" if torch.cuda.is_available() else "cpu",
            callbacks=[callback],
            fp16=args.fp16,
            checkpoint_cada=args.checkpoint_cada
        )
        
        # Guardar resultados del experimento
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        with open(os.path.join(dir_logs, f"experimento_{timestamp}.json"), 'w', encoding='utf-8') as f:
            json.dump({
                'parámetros': vars(args),
                'resultados': resultados,
                'textos_procesados': len(textos),
                'timestamp': timestamp
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Experimento completado y guardado en {dir_logs}")
        
        # Limpiar modelos si está configurado
        if args.max_modelos_guardados > 0:
            limpiar_modelos_anteriores(
                dir_modelos, 
                args.max_modelos_guardados, 
                args.guardar_solo_mejor
            )
    
    except Exception as e:
        logger.error(f"Error en el entrenamiento: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    try:
        start_time = time.time()
        main()
        logger.info(f"Proceso completado en {time.time() - start_time:.2f} segundos")
    except KeyboardInterrupt:
        logger.info("Proceso interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
