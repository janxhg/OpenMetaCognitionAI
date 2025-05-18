#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sistema Metacognitivo Integrado - Versión mejorada

Este módulo integra el sistema de aprendizaje autónomo con el sistema introspectivo,
combinando la capacidad de aprendizaje conceptual del sistema autónomo con
la capacidad de auto-modificación del sistema introspectivo para crear modelos de IA
rápidamente, fácilmente e inteligentes mediante metacognición avanzada.
"""

import os
import sys
import logging
import torch
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer

# Importar módulo para crear modelos desde cero y componentes básicos
from utils.crear_modelo_desde_cero import crear_modelo_y_tokenizador_nuevos, extraer_textos_de_materiales
from core.componentes_basicos import AutoObservacionBasica, ReflexionBasica, AutoModificacionBasica
from training.entrenamiento_metacognitivo import entrenar_modelo, IntrospectionCallback

# Configuración de paths para importaciones
import os
import sys

# Añadir rutas para los módulos
sistema_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(sistema_dir)
sys.path.append(parent_dir)

# Importar directamente de los archivos locales en el directorio integrado
try:
    # Importación directa de los archivos locales
    from cerebro_autonomo import CerebroAutonomo
    from aprendizaje_autonomo import MemoriaEstudio, BibliotecaConocimiento
    from auto_observacion import AutoObservacion
    from reflexion_metacognitiva import ReflexionMetacognitiva
    from auto_modificacion import AutoModificacion
    print("Importación directa de componentes metacognitivos exitosa")
except ImportError as e:
    print(f"Error al importar componentes metacognitivos locales: {str(e)}")
    print("Intentando importar desde rutas alternativas...")
    
    try:
        # Intentar con importaciones directas dentro de la carpeta core
        from cerebro_autonomo import CerebroAutonomo
        from aprendizaje_autonomo import MemoriaEstudio, BibliotecaConocimiento
        from auto_observacion import AutoObservacion
        from reflexion_metacognitiva import ReflexionMetacognitiva
        from auto_modificacion import AutoModificacion
        print("Importación alternativa de componentes metacognitivos exitosa")
    except ImportError as e:
        print(f"Error al importar componentes metacognitivos: {str(e)}")
        print("Usando implementaciones dummy para poder continuar")
        
        # Clases dummy para permitir que el sistema siga funcionando
        class CerebroAutonomo:
            def __init__(self, *args, **kwargs):
                print("ADVERTENCIA: Usando implementación dummy de CerebroAutonomo")
                self.modelo = None
                self.tokenizer = None
            
            def comprender_texto(self, texto):
                return "Comprensión simulada (dummy)"
                
            def extraer_conceptos(self, texto, dominio="general"):
                return {"concepto_dummy": {"definicion": "Definición simulada", "dominio": dominio}}
                
            def identificar_relaciones(self, concepto1, concepto2):
                return {"tipo": "relación_simulada", "fuerza": 5}
                
        class MemoriaEstudio:
            def __init__(self, *args, **kwargs):
                print("ADVERTENCIA: Usando implementación dummy de MemoriaEstudio")
                self.conceptos = {}
            
            def agregar_concepto(self, nombre, definicion, dominio="general", dificultad=5, importancia=5):
                self.conceptos[nombre] = {"definicion": definicion}
                
            def conectar_conceptos(self, concepto1, concepto2, tipo_relacion, fuerza=5):
                pass
                
            def cargar_memoria(self, ruta):
                pass
                
        class BibliotecaConocimiento:
            def __init__(self, *args, **kwargs):
                print("ADVERTENCIA: Usando implementación dummy de BibliotecaConocimiento")
                pass
                
        class AutoObservacion:
            def __init__(self, modelo, config=None):
                self.modelo = modelo
                self.config = config or {}
                print("Usando implementación dummy de AutoObservacion")
            
        class ReflexionMetacognitiva:
            def __init__(self, modelo=None, config=None):
                self.modelo = modelo
                self.config = config or {}
                print("Usando implementación dummy de ReflexionMetacognitiva")
                
        class AutoModificacion:
            def __init__(self, modelo=None, config=None):
                self.modelo = modelo
                self.config = config or {}
                print("Usando implementación dummy de AutoModificacion")
# Importar módulos de mejoras metacognitivas
try:
    from mejoras_metacognitivas.optimizacion_recursos import OptimizadorRecursos
    from mejoras_metacognitivas.introspeccion_avanzada import SistemaIntrospeccionAvanzada
    from mejoras_metacognitivas.auto_optimizacion import SistemaAutoOptimizacion
except ImportError:
    try:
        # Intentar con ruta relativa absoluta
        mejoras_path = os.path.join(sistema_dir, 'mejoras_metacognitivas')
        sys.path.append(mejoras_path)
        
        from optimizacion_recursos import OptimizadorRecursos
        from introspeccion_avanzada import SistemaIntrospeccionAvanzada
        from auto_optimizacion import SistemaAutoOptimizacion
    except ImportError as e:
        print(f"Error al importar mejoras metacognitivas: {str(e)}")
        print("Usando implementaciones dummy para mejoras")
        
        # Clases dummy para las mejoras
        class OptimizadorRecursos:
            def __init__(self, *args, **kwargs):
                pass
        
        class SistemaIntrospeccionAvanzada:
            def __init__(self, *args, **kwargs):
                pass
        
        class SistemaAutoOptimizacion:
            def __init__(self, *args, **kwargs):
                pass

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SistemaMetacognitivoIntegrado:
    """
    Sistema que integra las capacidades de aprendizaje autónomo e introspección.
    
    Combina:
    1. El Cerebro Autónomo para gestión conceptual y razonamiento
    2. La Memoria de Estudio para almacenar conocimiento estructurado
    3. La Auto-Observación para analizar los parámetros del modelo
    4. La Reflexión Metacognitiva para generar insights sobre el modelo
    5. La Auto-Modificación para implementar cambios en el modelo
    
    Esta versión mejorada proporciona:
    - Integración perfecta entre componentes autónomos e introspectivos
    - Ciclos de aprendizaje-reflexión-modificación acelerados
    - Interfaz simplificada para entrenar modelos rápidamente
    - Transferencia de conocimiento entre memoria y pesos del modelo
    - Evaluación y optimización automática del rendimiento
    """
    
    def __init__(self, 
                 modelo_path: str, 
                 dir_trabajo: str,
                 config: Dict = None,
                 modo_rapido: bool = False,
                 nivel_inteligencia: int = 5):
        """
        Inicializa el sistema metacognitivo integrado.
        
        Args:
            modelo_path: Ruta al modelo pre-entrenado o nombre del modelo en HuggingFace
            dir_trabajo: Directorio de trabajo para el sistema
            config: Configuración del sistema integrado
            modo_rapido: Si es True, optimiza para velocidad
            nivel_inteligencia: Nivel de inteligencia del sistema (1-10)
        """
        self.modelo_path = modelo_path
        self.dir_trabajo = dir_trabajo
        self.config = config or {}
        self.historial = []  # Historial de ciclos para registro
        
        # Configuración por defecto
        self.config.setdefault('max_modificacion_peso', 0.05)  # Limitar modificaciones al 5%
        self.config.setdefault('guardar_memoria', True)  # Guardar memoria de estudio después de cada ciclo
        self.config.setdefault('guardar_modelo', True)  # Guardar modelo después de cada ciclo
        self.config.setdefault('modo_rapido', modo_rapido)  # Modo rápido para entrenamiento acelerado
        self.config.setdefault('nivel_inteligencia', nivel_inteligencia)  # Nivel de inteligencia (1-10)
        self.config.setdefault('transferencia_conocimiento', True)  # Activar transferencia entre memoria y modelo
        self.config.setdefault('optimizacion_automatica', True)  # Optimizar automáticamente hiperparámetros
        
        # Configurar dispositivo para entrenar/inferencia
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Dispositivo seleccionado: {self.dispositivo}")

        
        # Crear directorios de trabajo
        os.makedirs(dir_trabajo, exist_ok=True)
        self.dir_memoria = os.path.join(dir_trabajo, "memoria")
        self.dir_modelos = os.path.join(dir_trabajo, "modelos")
        self.dir_logs = os.path.join(dir_trabajo, "logs")
        self.dir_optimizaciones = os.path.join(dir_trabajo, "optimizaciones")
        os.makedirs(self.dir_memoria, exist_ok=True)
        os.makedirs(self.dir_modelos, exist_ok=True)
        os.makedirs(self.dir_logs, exist_ok=True)
        os.makedirs(self.dir_optimizaciones, exist_ok=True)
        
        # Inicializar componentes básicos (algunos serán inicializados después de tener el modelo)
        self.cerebro = CerebroAutonomo()
        self.memoria_estudio = MemoriaEstudio()
        self.biblioteca = BibliotecaConocimiento()
        
        # Estos componentes requieren el modelo, los inicializaremos después
        self.auto_observacion = None
        self.reflexion = None
        self.auto_modificacion = None
        
        # Inicializar sistemas de mejora
        self.optimizador_recursos = OptimizadorRecursos()
        self.sistema_introspeccion = SistemaIntrospeccionAvanzada()
        self.sistema_optimizacion = SistemaAutoOptimizacion()
        self.crear_modelo_nuevo = modelo_path.lower() == "nuevo"
        
        # Inicializar modelo y tokenizer
        if self.crear_modelo_nuevo:
            # Configuración para el modelo nuevo a crear desde cero
            logger.info("Se creará un nuevo modelo desde cero")
            self.modelo = None
            self.tokenizer = None
            
            # Configuración para el modelo nuevo
            self.config_modelo_nuevo = {
                "vocab_size": 8000,  # Reducido para evitar problemas de memoria
                "n_positions": 256, 
                "n_ctx": 256,
                "n_embd": 192,
                "n_layer": 4,
                "n_head": 4,
            }
        else:
            # Cargar modelo existente
            logger.info(f"Cargando modelo desde {self.modelo_path}")
            try:
                # Primero verificamos si es una ruta local válida
                es_ruta_local = os.path.isdir(self.modelo_path) and os.path.exists(os.path.join(self.modelo_path, 'config.json'))
                
                if es_ruta_local:
                    # Es una ruta local, usamos un enfoque directo para evitar problemas con HF Hub
                    ruta_absoluta = os.path.abspath(self.modelo_path)
                    logger.info(f"Detectado modelo local en: {ruta_absoluta}")
                    
                    # Importación específica para modelos locales
                    from transformers import GPT2Config, GPT2LMHeadModel
                    import json
                    
                    # Verificar y cargar config.json
                    config_path = os.path.join(ruta_absoluta, "config.json")
                    if not os.path.exists(config_path):
                        raise ValueError(f"El archivo de configuración no existe: {config_path}")
                    
                    # Leer configuración manualmente
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                        
                    logger.info(f"Configuración cargada: {config_data.get('model_type', 'gpt2')}")
                    
                    # Crear objeto de configuración
                    config = GPT2Config(**config_data)
                    
                    # Verificar qué formato de archivo de pesos está disponible
                    tiene_pytorch = os.path.exists(os.path.join(ruta_absoluta, "pytorch_model.bin"))
                    tiene_safetensors = os.path.exists(os.path.join(ruta_absoluta, "model.safetensors"))
                    
                    if not (tiene_pytorch or tiene_safetensors):
                        raise ValueError(f"No se encontró ningún archivo de pesos en {ruta_absoluta}")
                    
                    # Cargar modelo con los parámetros adecuados para una ruta local
                    logger.info("Cargando pesos del modelo...")
                    use_safetensors = tiene_safetensors
                    self.modelo = GPT2LMHeadModel.from_pretrained(
                        ruta_absoluta,
                        config=config,
                        local_files_only=True,
                        use_safetensors=use_safetensors,
                        trust_remote_code=False
                    )
                    
                    # Cargar tokenizador de manera segura
                    logger.info("Cargando tokenizador...")
                    tokenizer_path = os.path.join(ruta_absoluta, "tokenizer.json")
                    if os.path.exists(tokenizer_path):
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            ruta_absoluta,
                            local_files_only=True,
                            use_fast=True
                        )
                        # Configurar pad_token para el tokenizador si no existe
                        if self.tokenizer.pad_token is None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token
                            logger.info("Configurando pad_token = eos_token para el tokenizador")
                    else:
                        # Cargar explícitamente el tokenizador especializado en español
                        ruta_tokenizador_español = "D:/AItrainer/Data/Tokenizers/Spanish/tokenizador_metacognitivo"
                        try:
                            logger.info(f"Intentando cargar tokenizador español metacognitivo desde: {ruta_tokenizador_español}")
                            self.tokenizer = AutoTokenizer.from_pretrained(ruta_tokenizador_español)
                            logger.info("Tokenizador español metacognitivo cargado correctamente")
                        except Exception as e:
                            logger.warning(f"No se pudo cargar el tokenizador español: {str(e)}")
                            logger.warning(f"No se encontró tokenizer.json en {ruta_absoluta}, usando tokenizador por defecto")
                            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                            
                        # Configurar pad_token para el tokenizador si no existe
                        if self.tokenizer.pad_token is None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token
                        
                    logger.info(f"Modelo local cargado exitosamente desde {ruta_absoluta}")
                else:
                    # Es un modelo de HuggingFace, carga normal
                    logger.info(f"Cargando modelo remoto desde HuggingFace: {self.modelo_path}")
                    self.modelo = AutoModelForCausalLM.from_pretrained(self.modelo_path)
                    self.tokenizer = AutoTokenizer.from_pretrained(self.modelo_path)
                    # Configurar pad_token para el tokenizador si no existe
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        logger.info("Configurando pad_token = eos_token para el tokenizador")
                    logger.info("Modelo remoto cargado exitosamente")
                
                # Mover modelo al dispositivo
                if self.modelo is not None:
                    self.modelo.to(self.dispositivo)
                    logger.info(f"Modelo cargado en dispositivo: {self.dispositivo}")
            except Exception as e:
                logger.error(f"Error al cargar el modelo: {str(e)}")
                raise e
        
        # Inicializar componentes autónomos con configuración mejorada
        try:
            logger.info(f"Inicializando cerebro autónomo con nivel de inteligencia {self.config['nivel_inteligencia']}")
            self.cerebro = CerebroAutonomo(modelo_path, self.dispositivo)
            self.memoria = MemoriaEstudio()
            self.biblioteca = BibliotecaConocimiento()
            
            # Configuración avanzada del cerebro autónomo
            if self.config['nivel_inteligencia'] > 7:
                logger.info("Activando capacidades avanzadas del cerebro autónomo")
                # Estas son propiedades que podríamos añadir al cerebro en el futuro
                # self.cerebro.activar_modo_creativo = True
                # self.cerebro.profundidad_razonamiento = self.config['nivel_inteligencia']
        except Exception as e:
            logger.warning(f"Error al inicializar componentes autónomos: {str(e)}. Usando configuración básica.")
            self.cerebro = CerebroAutonomo(modelo_path, self.dispositivo)
            self.memoria = MemoriaEstudio()
            self.biblioteca = BibliotecaConocimiento()
        
        # Cargar memoria si existe
        memoria_path = os.path.join(self.dir_memoria, "memoria_estudio.json")
        if os.path.exists(memoria_path):
            self.memoria.cargar_memoria(memoria_path)
            logger.info(f"Memoria de estudio cargada desde {memoria_path}")
        
        # Inicializar componentes introspectivos con configuración optimizada
        if self.modelo is not None:
            try:
                # Configuración para auto-observación
                config_observacion = {
                    'max_params_per_layer': 30,
                    'gradient_tracking': True,
                    'activation_tracking': True,
                    'histogram_bins': 10,
                    'max_layers_tracked': 8
                }
                self.auto_observacion = AutoObservacion(self.modelo, config_observacion)
                
                # Configuración para reflexión
                config_reflexion = {
                    'max_length': 512,
                    'temperatura': 0.7
                }
                self.reflexion = ReflexionMetacognitiva(self.modelo, self.tokenizer, self.auto_observacion, config_reflexion)
                
                # Configuración para auto-modificación
                self.auto_modificacion = AutoModificacion(
                    self.modelo, 
                    self.tokenizer, 
                    self.reflexion,
                    {'max_modificacion_peso': self.config['max_modificacion_peso']}
                )
                
                logger.info("Componentes introspectivos inicializados correctamente")
            except Exception as e:
                logger.warning(f"Error al inicializar componentes introspectivos: {str(e)}. Usando configuración básica.")
                # Intentar con configuración más básica
                self.auto_observacion = AutoObservacion(self.modelo)
                self.reflexion = ReflexionMetacognitiva(self.modelo, self.tokenizer, self.auto_observacion)
                self.auto_modificacion = AutoModificacion(self.modelo)
        else:
            # Si estamos creando modelo desde cero, dejamos los componentes como None por ahora
            self.auto_observacion = None
            self.reflexion = None
            self.auto_modificacion = None
            logger.info("Componentes introspectivos se inicializarán después de crear el modelo")
        
        # Historial del sistema integrado
        self.historial = []
        
        # Establecer puentes de transferencia de conocimiento si está activado
        if self.config['transferencia_conocimiento']:
            logger.info("Activando puentes de transferencia entre memoria y modelo")
            self._establecer_puentes_transferencia()
            
        # Optimizar configuración según el hardware disponible
        if self.config['optimizacion_automatica']:
            self._optimizar_para_hardware()
            
        logger.info("Sistema Metacognitivo Integrado inicializado con éxito")
    
    def inicializar_componentes_introspectivos(self):
        """
        Inicializa los componentes introspectivos (auto-observación, reflexión, auto-modificación)
        Esta función debe llamarse después de que el modelo esté disponible.
        """
        # Verificar que el modelo esté disponible
        if self.modelo is None:
            logger.error("No se puede inicializar componentes introspectivos sin un modelo disponible")
            return False
            
        try:
            logger.info("Inicializando componentes introspectivos...")
            
            # Inicializar componentes introspectivos
            self.auto_observacion = AutoObservacion(self.modelo)
            self.reflexion = ReflexionMetacognitiva(self.modelo, self.tokenizer, self.auto_observacion)
            self.auto_modificacion = AutoModificacion(self.modelo)
            
            # Asegurar que el cerebro autónomo también tenga acceso al modelo
            if hasattr(self, 'cerebro') and self.cerebro is not None:
                logger.info("Actualizando cerebro autónomo con el nuevo modelo")
                self.cerebro.modelo = self.modelo
                self.cerebro.tokenizer = self.tokenizer
            else:
                logger.info("Inicializando cerebro autónomo con el nuevo modelo")
                self.cerebro = CerebroAutonomo(self.modelo, self.tokenizer, self.dispositivo)
                
            logger.info("Componentes introspectivos y cerebro autónomo inicializados correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar componentes introspectivos: {str(e)}")
            return False
            
    def inicializar_modelo_desde_cero(self, materiales: List[Dict]) -> bool:
        """
        Inicializa un modelo y tokenizador desde cero usando textos de materiales.
        
        Args:
            materiales: Lista de materiales con textos para entrenar el tokenizador
            
        Returns:
            bool: True si el modelo se inicializó correctamente, False en caso contrario
        """
        if not self.crear_modelo_nuevo:
            return True
            
        if not materiales:
            logger.error("No hay materiales disponibles para crear el modelo desde cero.")
            return False
            
        try:
            # Extraer textos de los materiales
            textos = extraer_textos_de_materiales(materiales)
            if not textos:
                logger.error("No se pudieron extraer textos válidos de los materiales.")
                return False
                
            logger.info(f"Creando modelo desde cero con {len(textos)} textos de entrenamiento")
            
            # Crear modelo y tokenizador
            modelo_nuevo, tokenizador_nuevo = crear_modelo_y_tokenizador_nuevos(
                textos=textos,
                dir_salida=self.modelo_path,
                config_modelo=self.config_modelo_nuevo,
                vocab_size=self.config_modelo_nuevo.get("vocab_size", 8000)  # Valor reducido
            )
            
            # Verificar que el modelo y tokenizador se crearon correctamente
            if modelo_nuevo is None or tokenizador_nuevo is None:
                logger.error("No se pudo crear el modelo o el tokenizador")
                return False
                
            # Asignar modelo y tokenizador a la instancia
            self.modelo = modelo_nuevo
            self.tokenizer = tokenizador_nuevo
            
            # Mover modelo al dispositivo adecuado
            self.modelo.to(self.dispositivo)
            logger.info(f"Modelo nuevo creado y cargado en dispositivo: {self.dispositivo}")
            
            # Marcar que ya no es necesario crear un modelo nuevo
            self.crear_modelo_nuevo = False
            
            # Ahora que tenemos un modelo, inicializar los componentes introspectivos
            logger.info("Inicializando componentes introspectivos con el nuevo modelo...")
            self.inicializar_componentes_introspectivos()
            
            return True
            
        except Exception as e:
            logger.error(f"Error al crear modelo desde cero: {str(e)}")
            return False
        
    def ejecutar_ciclo_aprendizaje(self, 
                                  material_estudio: Dict,
                                  tiempo_estudio: int = 30,
                                  nivel_dificultad: int = 5) -> Dict:
        """
        Ejecuta un ciclo de aprendizaje autónomo sobre un material, incluyendo el
        entrenamiento real del modelo (fine-tuning) junto con la extracción de
        conceptos y reflexión metacognitiva.
        
        Args:
            material_estudio: Material de estudio (título, contenido, dominio)
            tiempo_estudio: Tiempo de estudio simulado en minutos
            nivel_dificultad: Nivel de dificultad del material (1-10)
            
        Returns:
            Resultados del ciclo de aprendizaje
        """
        logger.info(f"Iniciando ciclo de aprendizaje sobre {material_estudio['titulo']}")
        
        # Verificar que el modelo esté disponible
        if self.modelo is None:
            logger.error("No se puede ejecutar ciclo de aprendizaje sin un modelo disponible")
            return {
                'error': "Modelo no disponible",
                'timestamp': datetime.now().isoformat()
            }
        
        # FASE 1: ENTRENAMIENTO REAL DEL MODELO (FINE-TUNING)
        # Este paso entrena activamente el modelo con el material de estudio
        logger.info("Fase 1: Entrenamiento del modelo con el material de estudio")
        
        # Preparar contenido para entrenamiento
        contenido_texto = material_estudio['contenido']
        if isinstance(contenido_texto, str) and contenido_texto.strip():
            # Crear un callback de introspección para monitorear el entrenamiento
            introspection_callback = IntrospectionCallback(
                self.auto_observacion, 
                self.reflexion, 
                self.auto_modificacion
            )
            
            # Configurar parámetros de entrenamiento específicos para este ciclo
            # Usar valores ultra-conservadores para evitar problemas de CUDA OOM
            learning_rate = 2e-5
            num_epochs = 1 if self.config.get('modo_rapido', False) else 3
            batch_size = 1  # Batch ultra-pequeño para evitar OOM
            
            # Determinar la longitud máxima de secuencia según memoria disponible
            # Secuencias más cortas = menos uso de memoria
            max_length = 64  # Valor muy conservador por defecto
            
            # Detectar si disponemos de muy poca memoria CUDA
            memoria_limitada = False
            if self.dispositivo == "cuda" and torch.cuda.is_available():
                try:
                    memoria_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    memoria_libre_gb = torch.cuda.memory_reserved(0) / (1024**3)
                    
                    # Si tenemos menos de 4GB o más del 80% de la memoria ya está reservada
                    if memoria_total_gb < 4 or memoria_libre_gb / memoria_total_gb < 0.2:
                        memoria_limitada = True
                        logger.warning(f"Memoria GPU muy limitada: {memoria_libre_gb:.2f} GB libre de {memoria_total_gb:.2f} GB")
                    else:
                        # Con más memoria podemos tener secuencias un poco más largas
                        max_length = 96
                except Exception as e:
                    logger.warning(f"No se pudo determinar la memoria disponible: {str(e)}")
                    # Por seguridad asumimos memoria limitada
                    memoria_limitada = True
            
            # Aprendizaje en modo de contingencia si la memoria es muy limitada
            if memoria_limitada:
                # Limpiar caché de CUDA antes de entrenar
                torch.cuda.empty_cache()
                logger.info("Aplicando configuración ultra-conservadora para ahorrar memoria")
            
            # Entrenar el modelo con configuración optimizada para evitar OOM
            try:
                resultado_entrenamiento = entrenar_modelo(
                    modelo=self.modelo,
                    tokenizer=self.tokenizer,
                    textos=[contenido_texto],  # Usar el material de estudio actual
                    dir_salida=self.dir_modelos,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    estrategia_memoria="gradiente_acumulado",
                    max_length=max_length,  
                    dispositivo=self.dispositivo,
                    callbacks=[introspection_callback]
                )
                
                logger.info(f"Entrenamiento completado. Estado: {resultado_entrenamiento['estado']}")
                entrenamiento_exitoso = resultado_entrenamiento['estado'] == "completado"
                
                # Si hay error en el entrenamiento, registrarlo pero continuar con el ciclo
                if not entrenamiento_exitoso:
                    error_msg = resultado_entrenamiento.get('error', 'Error desconocido')
                    logger.warning(f"El entrenamiento no se completó correctamente: {error_msg}")
            except Exception as e:
                logger.error(f"Error durante el entrenamiento del modelo: {str(e)}")
                entrenamiento_exitoso = False
        else:
            logger.warning("Contenido no válido para entrenamiento del modelo")
            entrenamiento_exitoso = False
        
        # FASE 2: EXTRACCIÓN DE CONOCIMIENTO Y PROCESAMIENTO COGNITIVO
        # Ahora el modelo ya entrenado procesa conceptualmente el material
        logger.info("Fase 2: Extracción de conocimiento y procesamiento cognitivo")
        
        # 1. Comprender el texto del material
        significado = self.cerebro.comprender_texto(material_estudio['contenido'])
        
        # 2. Extraer conceptos del material
        conceptos_extraidos = self.cerebro.extraer_conceptos(
            material_estudio['contenido'], 
            material_estudio['dominio']
        )
        
        # Convertir el diccionario de conceptos a una lista para procesamiento posterior
        conceptos_lista = []
        for nombre, datos in conceptos_extraidos.items():
            concepto = {
                'nombre': nombre,
                'definicion': datos['definicion'],
                'dominio': datos.get('dominio', material_estudio['dominio']),
                'dificultad': datos.get('dificultad', nivel_dificultad),
                'importancia': datos.get('importancia', 5)
            }
            conceptos_lista.append(concepto)
        
        # 3. Agregar conceptos a la memoria
        for concepto in conceptos_lista:
            self.memoria.agregar_concepto(
                concepto['nombre'], 
                concepto['definicion'], 
                material_estudio['dominio'],
                concepto.get('dificultad', nivel_dificultad),
                concepto.get('importancia', 5)
            )
        
        # 4. Establecer relaciones entre conceptos
        for i, concepto1 in enumerate(conceptos_lista):
            for j, concepto2 in enumerate(conceptos_lista):
                if i < j:  # Evitar relaciones duplicadas y autorrelaciones
                    relacion = self.cerebro.identificar_relaciones(
                        concepto1['nombre'], 
                        concepto2['nombre']
                    )
                    if relacion:  # relacion es un diccionario único, no una lista
                        self.memoria.conectar_conceptos(
                            concepto1['nombre'],
                            concepto2['nombre'],
                            relacion['tipo'],  # Acceder directamente al diccionario
                            relacion.get('fuerza', 5)
                        )
        
        # 5. Generar preguntas sobre el material
        preguntas = self.cerebro.generar_preguntas(material_estudio['contenido'])
        
        # FASE 3: REFLEXIÓN METACOGNITIVA Y ALMACENAMIENTO
        logger.info("Fase 3: Reflexión metacognitiva y almacenamiento")
        
        # 6. Reflexionar sobre el aprendizaje (ahora incluye el entrenamiento real)
        info_entrenamiento = f"Entrenamiento del modelo: {'Exitoso' if entrenamiento_exitoso else 'Fallido'}. Se extrajeron {len(conceptos_lista)} conceptos del material."
        logger.info(info_entrenamiento)
    
        # No podemos pasar reflexion_adicional directamente, así que la concatenamos al prompt manualmente
        # Primero obtenemos la reflexión estándar
        reflexion_aprendizaje = self.cerebro.reflexionar_sobre_aprendizaje(
            [c['nombre'] for c in conceptos_lista],
            tiempo_estudio
        )
        
        # Luego añadimos nuestra información adicional de forma manual
        reflexion_aprendizaje = f"{reflexion_aprendizaje}\n\nInformación adicional: {info_entrenamiento}"
        
        # 7. Registrar sesión de estudio en la memoria
        calidad_sesion = 8 if entrenamiento_exitoso else 5  # Mejor calidad si el entrenamiento fue exitoso
        self.memoria.registrar_sesion_estudio(
            material_estudio['titulo'],
            tiempo_estudio,
            [c['nombre'] for c in conceptos_lista],
            calidad_sesion
        )
        
        # 8. Agregar reflexión a la memoria
        self.memoria.agregar_reflexion(reflexion_aprendizaje)
        
        # 9. Guardar memoria si está configurado
        if self.config['guardar_memoria']:
            memoria_path = os.path.join(self.dir_memoria, "memoria_estudio.json")
            self.memoria.guardar_memoria(memoria_path)
        
        # 10. Guardar el modelo actualizado si está configurado
        if self.config['guardar_modelo'] and entrenamiento_exitoso:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            modelo_path = os.path.join(self.dir_modelos, f"modelo_entrenado_{timestamp}")
            os.makedirs(modelo_path, exist_ok=True)
            
            logger.info(f"Guardando modelo actualizado en {modelo_path}")
            self.modelo.save_pretrained(modelo_path)
            self.tokenizer.save_pretrained(modelo_path)
        
        # Preparar resultados del ciclo
        resultados = {
            'material': material_estudio['titulo'],
            'dominio': material_estudio['dominio'],
            'entrenamiento': {
                'exitoso': entrenamiento_exitoso,
                'detalles': resultado_entrenamiento if entrenamiento_exitoso else {'error': 'Entrenamiento fallido o no realizado'}
            },
            'conceptos_extraidos': conceptos_lista,
            'preguntas_generadas': preguntas,
            'reflexion': reflexion_aprendizaje,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Ciclo de aprendizaje completado. {len(conceptos_lista)} conceptos extraídos.")
        return resultados
    
    def _verificar_componentes_introspectivos(self) -> bool:
        """
        Verifica si los componentes introspectivos están inicializados correctamente.
        Si no lo están, intenta inicializarlos de manera segura.
        
        Returns:
            bool: True si los componentes están disponibles y funcionando
        """
        # Verificar si tenemos un modelo válido
        if self.modelo is None:
            logger.error("No se puede realizar introspección sin un modelo válido")
            return False
            
        # Verificar si alguno de los componentes es None
        componentes_faltantes = []
        if self.auto_observacion is None:
            componentes_faltantes.append("auto_observacion")
        if self.reflexion is None:
            componentes_faltantes.append("reflexion")
        if self.auto_modificacion is None:
            componentes_faltantes.append("auto_modificacion")
            
        if componentes_faltantes:
            logger.warning(f"Componentes faltantes: {', '.join(componentes_faltantes)}. Intentando inicializar...")
            try:
                # Intentar inicializar los componentes faltantes uno por uno
                if "auto_observacion" in componentes_faltantes:
                    self.auto_observacion = AutoObservacion(self.modelo)
                    logger.info("Auto-observación inicializada")
                
                # Inicializar reflexión si falta (depende de auto_observacion)
                if "reflexion" in componentes_faltantes:
                    if self.auto_observacion is not None:
                        self.reflexion = ReflexionMetacognitiva(self.modelo, self.tokenizer, self.auto_observacion)
                        logger.info("Reflexión metacognitiva inicializada")
                    else:
                        # Usar versión básica si auto_observacion no está disponible
                        self.reflexion = ReflexionBasica(self.modelo, self.tokenizer)
                        logger.info("Usando reflexión básica")
                
                # Inicializar auto_modificacion si falta
                if "auto_modificacion" in componentes_faltantes:
                    self.auto_modificacion = AutoModificacion(self.modelo, self.tokenizer, self.reflexion)
                    logger.info("Auto-modificación inicializada")
                    
                return True
            except Exception as e:
                logger.error(f"Error al inicializar componentes: {str(e)}")
                # Crear versiones básicas como último recurso
                logger.warning("Usando componentes básicos de emergencia")
                if "auto_observacion" in componentes_faltantes:
                    self.auto_observacion = AutoObservacionBasica(self.modelo)
                if "reflexion" in componentes_faltantes:
                    self.reflexion = ReflexionBasica(self.modelo, self.tokenizer)
                if "auto_modificacion" in componentes_faltantes:
                    self.auto_modificacion = AutoModificacionBasica(self.modelo, self.tokenizer)
                return True
        
        # Todos los componentes están inicializados
        return True
            
    def ejecutar_ciclo_introspectivo(self, datos_adicionales: Dict = None) -> Dict:
        """
        Ejecuta un ciclo de introspección y auto-modificación del modelo.
        
        Args:
            datos_adicionales: Datos adicionales para la reflexión
            
        Returns:
            Resultados del ciclo introspectivo
        """
        logger.info("Iniciando ciclo introspectivo")
        
        # Verificar que los componentes introspectivos estén disponibles
        if not self._verificar_componentes_introspectivos():
            logger.error("No se puede ejecutar el ciclo introspectivo sin los componentes necesarios")
            return {
                'error': "Componentes introspectivos no disponibles",
                'timestamp': datetime.now().isoformat()
            }
        
        # 1. Observar el estado actual del modelo
        self.auto_observacion.observar_pesos()
        
        # 2. Enriquecer datos adicionales con memoria de estudio
        if datos_adicionales is None:
            datos_adicionales = {}
        
        # Añadir información sobre conceptos y sesiones de la memoria
        datos_adicionales['num_conceptos'] = len(self.memoria.conceptos)
        datos_adicionales['num_conexiones'] = len(self.memoria.conexiones)
        datos_adicionales['num_sesiones'] = len(self.memoria.sesiones_estudio)
        
        # Añadir últimas reflexiones de la memoria
        if self.memoria.reflexiones:
            datos_adicionales['reflexiones_previas'] = self.memoria.reflexiones[-3:]
        
        # 3. Generar reflexión metacognitiva
        reflexion = self.reflexion.generar_reflexion(datos_adicionales)
        
        # 4. Analizar problemas de aprendizaje
        problemas = self.reflexion.analizar_problemas_aprendizaje()
        
        # 5. Generar estrategias de mejora
        estrategias = self.reflexion.generar_estrategias_mejora(problemas)
        
        # 6. Ejecutar ciclo de auto-modificación
        modificaciones = self.auto_modificacion.ejecutar_ciclo_metacognitivo({
            'reflexion': reflexion,
            'problemas': problemas,
            'estrategias': estrategias,
            'datos_memoria': {
                'conceptos': list(self.memoria.conceptos.keys()),
                'sesiones': len(self.memoria.sesiones_estudio)
            }
        })
        
        # 7. Guardar modelo modificado si está configurado
        if self.config['guardar_modelo']:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            modelo_dir = os.path.join(self.dir_modelos, f"modelo_modificado_{timestamp}")
            os.makedirs(modelo_dir, exist_ok=True)
            self.modelo.save_pretrained(modelo_dir)
            self.tokenizer.save_pretrained(modelo_dir)
            logger.info(f"Modelo modificado guardado en {modelo_dir}")
        
        resultados = {
            'reflexion': reflexion,
            'problemas': problemas,
            'estrategias': estrategias,
            'modificaciones': modificaciones,
            'timestamp': datetime.now().isoformat()
        }
        
        # Guardar resultados en historial
        self.historial.append(resultados)
        
        logger.info("Ciclo introspectivo completado")
        return resultados
    
    def ejecutar_ciclo_completo(self, 
                               material_estudio: Dict = None,
                               datos_adicionales: Dict = None) -> Dict:
        """
        Ejecuta un ciclo completo de aprendizaje e introspección con mejoras metacognitivas.
        
        Args:
            material_estudio: Material de estudio (opcional)
            datos_adicionales: Datos adicionales para la reflexión
            
        Returns:
            Resultados del ciclo completo con mejoras metacognitivas
        """
        logger.info("Iniciando ciclo completo con mejoras metacognitivas")
        
        resultados = {}
        
        # 1. Optimización de recursos inicial
        self.optimizador_recursos.optimizar_recursos(self.modelo)
        
        # 2. Si hay material de estudio, ejecutar ciclo de aprendizaje
        if material_estudio:
            resultados['aprendizaje'] = self.ejecutar_ciclo_aprendizaje(material_estudio)
            
            # Actualizar datos adicionales con resultados del aprendizaje
            if datos_adicionales is None:
                datos_adicionales = {}
            datos_adicionales['ultimo_aprendizaje'] = {
                'material': material_estudio['titulo'],
                'conceptos': [c['nombre'] for c in resultados['aprendizaje']['conceptos_extraidos']],
                'reflexion': resultados['aprendizaje']['reflexion']
            }
            
            # 3. Introspección avanzada
            reporte_introspeccion = self.sistema_introspeccion.analizar_modelo(
                self.modelo, 
                torch.tensor([self.tokenizer.encode(material_estudio['contenido'])[:512]])
            )
            datos_adicionales['reporte_introspeccion'] = reporte_introspeccion
            
            # 4. Optimización automática
            hiperparametros = self.sistema_optimizacion.optimizar_entrenamiento(
                self.modelo, 
                self._preparar_datos_validacion(material_estudio)
            )
            datos_adicionales['hiperparametros_optimizados'] = hiperparametros
            
            # 5. Guardar estados de optimización
            self.optimizador_recursos.guardar_estado(
                os.path.join(self.dir_optimizaciones, f"optimizador_{datetime.now().strftime('%Y%m%d%H%M%S')}.pth")
            )
            self.sistema_optimizacion.guardar_estado(
                os.path.join(self.dir_optimizaciones, f"optimizacion_{datetime.now().strftime('%Y%m%d%H%M%S')}.pth")
            )
            
        # 6. Ejecutar ciclo introspectivo con mejoras
        resultados['introspectivo'] = self.ejecutar_ciclo_introspectivo(datos_adicionales)
        
        # 7. Generar reportes adicionales
        if material_estudio:
            reporte_optimizacion = self.sistema_optimizacion.generar_reporte_optimizacion()
            reporte_introspeccion = self.sistema_introspeccion.generar_reporte_introspeccion()
            
            resultados['reportes'] = {
                'optimizacion': reporte_optimizacion,
                'introspeccion': reporte_introspeccion
            }
            
        return resultados
        
    def _preparar_datos_validacion(self, material_estudio: Dict) -> List[Dict]:
        """
        Prepara datos de validación para la optimización.
        
        Args:
            material_estudio: Material de estudio para crear los datos de validación
            
        Returns:
            Lista de diccionarios con datos de validación
        """
        # Preparar entrada de ejemplo
        entrada = torch.tensor([self.tokenizer.encode(material_estudio['contenido'])[:512]])
        
        # Generar salida de referencia
        self.modelo.eval()
        with torch.no_grad():
            salida = self.modelo(entrada)
        
        return [{'input': entrada, 'target': salida}]
        
        # 2. Ejecutar ciclo introspectivo
        resultados['introspectivo'] = self.ejecutar_ciclo_introspectivo(datos_adicionales)
        
        # 3. Guardar resultados completos
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        with open(os.path.join(self.dir_logs, f"ciclo_{timestamp}.json"), "w", encoding="utf-8") as f:
            json.dump(resultados, f, ensure_ascii=False, indent=2)
            
        logger.info("Ciclo completo finalizado")
        return resultados
    
    def ejecutar_ciclos_multiples(self, 
                                materiales: List[Dict],
                                num_ciclos: int = 3) -> List[Dict]:
        """
        Ejecuta múltiples ciclos de aprendizaje e introspección.
        
        Args:
            materiales: Lista de materiales de estudio
            num_ciclos: Número de ciclos a ejecutar
            
        Returns:
            Lista con resultados de todos los ciclos
        """
        logger.info(f"Iniciando ejecución de {num_ciclos} ciclos")
        
        resultados_ciclos = []
        
        # Si es un modelo nuevo, inicializarlo primero
        if self.crear_modelo_nuevo:
            self.inicializar_modelo_desde_cero(materiales)
            
        # Verificar que los componentes introspectivos estén disponibles
        self._verificar_componentes_introspectivos()
        
        # Verificar que hay materiales disponibles
        if not materiales:
            logger.warning("No hay materiales disponibles para el entrenamiento")
            return resultados_ciclos
        
        # Preparar un "material combinado" que contenga todos los textos de entrenamiento
        # para usarlo en cada ciclo, maximizando el aprovechamiento de datos
        logger.info(f"Preparando conjunto combinado con TODOS los {len(materiales)} textos disponibles")
        
        # Determinar longitud máxima combinada para evitar problemas de memoria
        max_textos_por_ciclo = min(50, len(materiales))  # Límite razonable
        
        # Crear lotes de materiales para procesar eficientemente
        lotes_materiales = []
        for i in range(0, len(materiales), max_textos_por_ciclo):
            lote = materiales[i:i + max_textos_por_ciclo]
            lotes_materiales.append(lote)
            
        logger.info(f"Datos divididos en {len(lotes_materiales)} lotes para procesamiento eficiente")
            
        for i in range(num_ciclos):
            logger.info(f"Ejecutando ciclo {i+1}/{num_ciclos}")
            
            # Para cada ciclo, procesamos todos los lotes de materiales
            resultados_ciclo = {}
            
            for j, lote in enumerate(lotes_materiales):
                logger.info(f"Procesando lote {j+1}/{len(lotes_materiales)} con {len(lote)} textos")
                
                # Ejecutar ciclo completo con este lote de materiales
                # Creamos un material artificial que contiene todos los textos del lote
                material_combinado = {
                    'titulo': f"conjunto_entrenamiento_lote_{j+1}_ciclo_{i+1}",
                    'contenido': '\n\n'.join([m['contenido'] for m in lote]),
                    'dominio': 'combinado',
                    'dificultad': self.config['nivel_inteligencia']
                }
                
                # Ejecutar ciclo completo con todos los textos combinados
                resultados_lote = self.ejecutar_ciclo_completo(material_combinado)
                
                # Actualizar resultados del ciclo con los de este lote
                if not resultados_ciclo:
                    resultados_ciclo = resultados_lote
                else:
                    # Combinar los resultados si es necesario
                    if 'aprendizaje' in resultados_lote and 'conceptos_extraidos' in resultados_lote['aprendizaje']:
                        if 'aprendizaje' not in resultados_ciclo:
                            resultados_ciclo['aprendizaje'] = {}
                        if 'conceptos_extraidos' not in resultados_ciclo['aprendizaje']:
                            resultados_ciclo['aprendizaje']['conceptos_extraidos'] = []
                        
                        resultados_ciclo['aprendizaje']['conceptos_extraidos'].extend(
                            resultados_lote['aprendizaje'].get('conceptos_extraidos', [])
                        )
            
            # Agregar los resultados combinados de este ciclo
            resultados_ciclos.append(resultados_ciclo)
            
            # Actualizar estado para el siguiente ciclo
            estado_sistema = self.evaluar_estado_sistema()
            logger.info(f"Estado del sistema tras ciclo {i+1}: {estado_sistema}")
            
            # Opcional: pausar entre ciclos para permitir enfriamiento de GPU
            if i < num_ciclos - 1:
                import time
                time.sleep(5)
        
        # Generar y guardar informe final
        self._generar_informe_final(resultados_ciclos)
        
        logger.info(f"Completados {num_ciclos} ciclos")
        return resultados_ciclos
    
    def evaluar_estado_sistema(self) -> Dict:
        """
        Evalúa el estado actual del sistema integrado.
        
        Returns:
            Diccionario con métricas del estado del sistema
        """
        # Recopilar estadísticas de la memoria
        memoria_stats = {
            'num_conceptos': len(self.memoria.conceptos),
            'num_conexiones': len(self.memoria.conexiones),
            'num_sesiones': len(self.memoria.sesiones_estudio),
            'num_reflexiones': len(self.memoria.reflexiones)
        }
        
        # Evaluar conocimiento en algunos dominios
        dominios = set()
        for concepto in self.memoria.conceptos.values():
            dominios.add(concepto['dominio'])
        
        evaluaciones_dominio = {}
        try:
            for dominio in list(dominios)[:3]:  # Limitar a 3 dominios para eficiencia
                evaluacion = self.cerebro.autoevaluar_conocimiento(dominio)
                evaluaciones_dominio[dominio] = evaluacion
        except Exception as e:
            logger.warning(f"Error al evaluar dominios: {str(e)}")
            # Agregar al menos una evaluación por defecto para evitar errores posteriores
            evaluaciones_dominio['desconocido'] = {
                'texto': f"Error: {str(e)}",
                'nivel': 'principiante'
            }
        
        # Información sobre modificaciones realizadas
        modificaciones_info = {
            'num_ciclos_introspectivos': len(self.historial),
            'total_modificaciones': 0  # Valor por defecto
        }
        
        # Calcular total de modificaciones de manera segura
        if self.historial:
            try:
                total_mods = 0
                for ciclo in self.historial:
                    if 'modificaciones' in ciclo:
                        mods = ciclo['modificaciones']
                        if isinstance(mods, dict) and 'resultados_modificaciones' in mods:
                            res_mods = mods['resultados_modificaciones']
                            if isinstance(res_mods, dict) and 'modificaciones_aplicadas' in res_mods:
                                total_mods += len(res_mods['modificaciones_aplicadas'])
                                
                modificaciones_info['total_modificaciones'] = total_mods
            except Exception as e:
                logger.warning(f"Error al contar modificaciones: {str(e)}")
        
        return {
            'memoria': memoria_stats,
            'evaluaciones_dominio': evaluaciones_dominio,
            'modificaciones': modificaciones_info,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generar_informe_final(self, resultados_ciclos: List[Dict]):
        """
        Genera y guarda un informe final de todos los ciclos ejecutados.
        
        Args:
            resultados_ciclos: Lista con resultados de todos los ciclos
        """
        logger.info("Generando informe final")
        
        # Evaluar estado final del sistema
        estado_final = self.evaluar_estado_sistema()
        
        # Generar síntesis de conocimiento
        sintesis = self.cerebro.sintetizar_conocimiento(self.memoria.conceptos)
        
        # Crear informe
        informe = {
            'fecha': datetime.now().isoformat(),
            'modelo_base': self.modelo_path,
            'num_ciclos_ejecutados': len(resultados_ciclos),
            'estado_final': estado_final,
            'sintesis_conocimiento': sintesis,
            'resumen_ciclos': []
        }
        
        # Agregar resumen de cada ciclo
        for i, ciclo in enumerate(resultados_ciclos):
            resumen_ciclo = {
                'ciclo_num': i + 1,
                'timestamp': ciclo.get('timestamp', '')
            }
            
            if 'aprendizaje' in ciclo:
                resumen_ciclo['material_estudiado'] = ciclo['aprendizaje'].get('material', '')
                resumen_ciclo['conceptos_extraidos'] = len(ciclo['aprendizaje'].get('conceptos_extraidos', []))
            
            if 'introspectivo' in ciclo and 'modificaciones' in ciclo['introspectivo']:
                mod_data = ciclo['introspectivo']['modificaciones'].get('resultados_modificaciones', {})
                resumen_ciclo['modificaciones_aplicadas'] = len(mod_data.get('modificaciones_aplicadas', []))
                resumen_ciclo['modificaciones_rechazadas'] = len(mod_data.get('modificaciones_rechazadas', []))
            
            informe['resumen_ciclos'].append(resumen_ciclo)
        
        # Guardar informe
        with open(os.path.join(self.dir_trabajo, "informe_final.json"), "w", encoding="utf-8") as f:
            json.dump(informe, f, ensure_ascii=False, indent=2)
        
        # Versión legible
        with open(os.path.join(self.dir_trabajo, "informe_final.txt"), "w", encoding="utf-8") as f:
            f.write("INFORME FINAL DEL SISTEMA METACOGNITIVO INTEGRADO\n")
            f.write("=================================================\n\n")
            f.write(f"Fecha: {informe['fecha']}\n")
            f.write(f"Modelo base: {informe['modelo_base']}\n")
            f.write(f"Ciclos ejecutados: {informe['num_ciclos_ejecutados']}\n\n")
            
            f.write("ESTADO FINAL DEL SISTEMA\n")
            f.write("========================\n")
            f.write(f"Conceptos en memoria: {estado_final['memoria']['num_conceptos']}\n")
            f.write(f"Conexiones entre conceptos: {estado_final['memoria']['num_conexiones']}\n")
            f.write(f"Sesiones de estudio: {estado_final['memoria']['num_sesiones']}\n")
            f.write(f"Reflexiones acumuladas: {estado_final['memoria']['num_reflexiones']}\n")
            f.write(f"Total modificaciones aplicadas: {estado_final['modificaciones']['total_modificaciones']}\n\n")
            
            f.write("SÍNTESIS DE CONOCIMIENTO\n")
            f.write("=======================\n")
            f.write(f"{sintesis}\n\n")
            
            f.write("RESUMEN DE CICLOS\n")
            f.write("================\n")
            for resumen in informe['resumen_ciclos']:
                f.write(f"Ciclo {resumen['ciclo_num']}:\n")
                if 'material_estudiado' in resumen:
                    f.write(f"  Material: {resumen['material_estudiado']}\n")
                if 'conceptos_extraidos' in resumen:
                    f.write(f"  Conceptos extraídos: {resumen['conceptos_extraidos']}\n")
                if 'modificaciones_aplicadas' in resumen:
                    f.write(f"  Modificaciones aplicadas: {resumen['modificaciones_aplicadas']}\n")
                    f.write(f"  Modificaciones rechazadas: {resumen['modificaciones_rechazadas']}\n")
                f.write("\n")
        
        logger.info(f"Informe final guardado en {self.dir_trabajo}")


# Función auxiliar para crear una instancia del sistema
    def _establecer_puentes_transferencia(self):
        """Establece puentes de transferencia entre la memoria conceptual y los pesos del modelo."""
        try:
            # Esta función simula una conexión entre el conocimiento conceptual y los parámetros del modelo
            logger.info("Estableciendo puentes de transferencia conocimiento-parámetros")
            # En una implementación real, aquí podríamos añadir mecanismos para que
            # el conocimiento en la memoria influya en cómo se modifican los pesos
        except Exception as e:
            logger.warning(f"Error al establecer puentes de transferencia: {str(e)}")
    
    def _optimizar_para_hardware(self):
        """Optimiza la configuración basándose en el hardware disponible."""
        try:
            # Detectar CUDA y memoria disponible
            if torch.cuda.is_available():
                cuda_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                logger.info(f"Optimizando para CUDA con {cuda_mem:.2f} GB de memoria")
                
                # Ajustar configuración según memoria disponible
                if cuda_mem < 4:
                    logger.info("Detectada GPU con memoria limitada. Ajustando parámetros para rendimiento óptimo.")
                    self.config['modo_rapido'] = True
                    if hasattr(self.cerebro, 'configurar_modo_ligero'):
                        self.cerebro.configurar_modo_ligero(True)
        except Exception as e:
            logger.warning(f"Error al optimizar para hardware: {str(e)}")

    def entrenar_rapido(self, materiales, num_ciclos=3, nombre_modelo="modelo_metacognitivo_rapido"):
        """Método conveniente para entrenar un modelo rápidamente con configuración optimizada.
        
        Args:
            materiales: Lista de materiales de estudio
            num_ciclos: Número de ciclos a ejecutar
            nombre_modelo: Nombre para el modelo resultante
            
        Returns:
            Ruta al modelo entrenado
        """
        logger.info(f"Iniciando entrenamiento rápido con {num_ciclos} ciclos")
        
        # Activar modo rápido si no está ya activado
        modo_rapido_original = self.config['modo_rapido']
        self.config['modo_rapido'] = True
        
        # Ejecutar ciclos
        resultados = self.ejecutar_ciclos_multiples(materiales, num_ciclos)
        
        # Guardar modelo final con nombre personalizado
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        ruta_modelo = os.path.join(self.dir_modelos, f"{nombre_modelo}_{timestamp}")
        os.makedirs(ruta_modelo, exist_ok=True)
        
        # Verificar que el modelo y tokenizer estén disponibles antes de guardar
        if self.modelo is None:
            logger.error("No se puede guardar el modelo porque no está inicializado")
            return None
            
        try:
            self.modelo.save_pretrained(ruta_modelo)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(ruta_modelo)
            else:
                logger.warning("No se pudo guardar el tokenizer porque no está inicializado")
        except Exception as e:
            logger.error(f"Error al guardar el modelo: {str(e)}")
            return None
        
        # Restaurar configuración original
        self.config['modo_rapido'] = modo_rapido_original
        
        logger.info(f"Entrenamiento rápido completado. Modelo guardado en {ruta_modelo}")
        return ruta_modelo


def crear_sistema_integrado(modelo_path, dir_trabajo, config=None, modo_rapido=False, nivel_inteligencia=5, parametros_entrenamiento=None):
    """
    Crea y configura un sistema metacognitivo integrado.
    
    Args:
        modelo_path: Ruta al modelo pre-entrenado o nombre del modelo en HuggingFace
        dir_trabajo: Directorio de trabajo para el sistema
        config: Configuración opcional
        modo_rapido: Si es True, utiliza configuración optimizada para velocidad
        nivel_inteligencia: Nivel de inteligencia del sistema (1-10)
        parametros_entrenamiento: Diccionario con parámetros adicionales para el entrenamiento
            - batch_size: Tamaño de lote para entrenamiento
            - learning_rate: Tasa de aprendizaje para el optimizador
            - num_epochs: Número de épocas por ciclo de entrenamiento
            - checkpointing: Si se deben guardar checkpoints durante el entrenamiento
            - max_grad_norm: Norma máxima de gradiente para gradient clipping
            - warmup_steps: Pasos de calentamiento para el scheduler
            - weight_decay: Weight decay para el optimizador
        
    Returns:
        Instancia de SistemaMetacognitivoIntegrado
    """
    # Si hay parámetros de entrenamiento, agregarlos a la configuración
    if parametros_entrenamiento:
        if config is None:
            config = {}
        config['parametros_entrenamiento'] = parametros_entrenamiento
        
    return SistemaMetacognitivoIntegrado(modelo_path, dir_trabajo, config, modo_rapido, nivel_inteligencia)


if __name__ == "__main__":
    # Ejemplo de uso mejorado
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Usar un modelo pequeño para pruebas
    modelo_nombre = "MetaCognitive"
    dir_trabajo_temp = "./sistema_metacognitivo_demo"
    
    # Crear el sistema integrado con configuración optimizada
    sistema = crear_sistema_integrado(
        modelo_path=modelo_nombre, 
        dir_trabajo=dir_trabajo_temp,
        modo_rapido=True,  # Usar modo rápido para demo
        nivel_inteligencia=7  # Nivel de inteligencia alto
    )
    
    # Material de estudio de ejemplo
    material_ejemplo = {
        'titulo': 'Introducción a la Metacognición',
        'contenido': """
        La metacognición se refiere al conocimiento y regulación de nuestros propios procesos cognitivos.
        Implica ser consciente de cómo pensamos y aprendemos, y ser capaz de controlar y ajustar estos procesos.
        Los componentes principales de la metacognición incluyen:
        1. Conocimiento metacognitivo: saber sobre nuestras propias capacidades cognitivas.
        2. Regulación metacognitiva: planificación, monitoreo y evaluación de nuestro aprendizaje.
        3. Experiencias metacognitivas: sensaciones y experiencias que acompañan a las actividades cognitivas.
        
        La metacognición es crucial para el aprendizaje efectivo y la resolución de problemas.
        Permite a los estudiantes identificar estrategias apropiadas, monitorear su comprensión,
        y ajustar su enfoque cuando encuentran dificultades.
        """,
        'dominio': 'Psicología Cognitiva',
        'dificultad': 5
    }
    
    # Ejecutar un ciclo completo
    resultados = sistema.ejecutar_ciclo_completo(material_ejemplo)
    
    # Mostrar algunos resultados
    print("\nCICLO COMPLETO EJECUTADO")
    print("========================")
    print(f"Conceptos extraídos: {len(resultados['aprendizaje']['conceptos_extraidos'])}")
    print(f"Preguntas generadas: {len(resultados['aprendizaje']['preguntas_generadas'])}")
    
    if 'modificaciones' in resultados['introspectivo']:
        mod_data = resultados['introspectivo']['modificaciones'].get('resultados_modificaciones', {})
        print(f"Modificaciones aplicadas: {len(mod_data.get('modificaciones_aplicadas', []))}")
        print(f"Modificaciones rechazadas: {len(mod_data.get('modificaciones_rechazadas', []))}")
    
    print("\nESTADO DEL SISTEMA")
    print("=================")
    estado = sistema.evaluar_estado_sistema()
    print(f"Conceptos en memoria: {estado['memoria']['num_conceptos']}")
    print(f"Conexiones entre conceptos: {estado['memoria']['num_conexiones']}")
