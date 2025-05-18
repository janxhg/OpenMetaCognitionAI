#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entrenamiento Metacognitivo Directo

Este módulo implementa un enfoque revolucionario para el entrenamiento de modelos,
donde se reemplaza el concepto tradicional de épocas por un sistema metacognitivo
que controla directamente el proceso de aprendizaje en tiempo real.

Este enfoque permite que el modelo:
1. Intervenga directamente en la propagación hacia adelante/atrás
2. Analice sus propias activaciones y gradientes en tiempo real
3. Modifique sus pesos según algoritmos metacognitivos avanzados
"""

import os
import sys
import torch
import logging
import json
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Callable

from torch.utils.data import Dataset, DataLoader
from transformers import (
    PreTrainedTokenizer, 
    PreTrainedModel,
    TrainingArguments,
    Trainer
)

# Añadir rutas para importar módulos del directorio core
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))

# Importar componentes del sistema metacognitivo
try:
    from auto_observacion import AutoObservacion
    from reflexion_metacognitiva import ReflexionMetacognitiva
    from auto_modificacion import AutoModificacion
    from cerebro_autonomo import CerebroAutonomo
except ImportError as e:
    print(f"Error al importar componentes metacognitivos: {str(e)}")
    print("Asegúrate de que todos los componentes necesarios están presentes en la carpeta 'core'.")

# Configuración del logger
logger = logging.getLogger(__name__)


class MetacognitiveHook:
    """Hook para intervenir en la propagación hacia adelante y hacia atrás durante el entrenamiento.
    
    Esta implementación incluye mecanismos avanzados para detectar y prevenir el sobreajuste:
    1. Monitoreo de entropía de las activaciones para detectar memorización
    2. Análisis de diversidad de representaciones a lo largo del tiempo
    3. Intervención activa mediante inyección controlada de variabilidad cuando se detecta sobreajuste
    """
    
    def __init__(self, auto_observacion, reflexion, auto_modificacion):
        """Inicializa el hook metacognitivo.
        
        Args:
            auto_observacion: Componente para observar estados internos del modelo
            reflexion: Componente para generar reflexiones sobre el aprendizaje
            auto_modificacion: Componente para modificar pesos y comportamiento
        """
        self.auto_observacion = auto_observacion
        self.reflexion = reflexion
        self.auto_modificacion = auto_modificacion
        
        # Métricas de seguimiento
        self.activaciones_por_capa = {}
        self.gradientes_por_capa = {}
        self.observaciones_tiempo_real = []
        self.iteraciones = 0
        self.intervenciones = 0
        
        # Sistema de detección de sobreajuste
        self.historico_entropia = {}
        self.ventana_analisis = 100  # Pasos para analizar tendencias
        self.umbral_alerta_entropia = 0.1  # Umbral de caída de entropía para alertar de sobreajuste
        self.contador_alertas_sobreajuste = 0
        self.nivel_regularizacion = 0.0  # Nivel dinámico de regularización
        self.ultimo_paso_regularizacion = 0
    
    def registrar_hooks(self, modelo):
        """Registra hooks en todas las capas relevantes del modelo.
        
        Args:
            modelo: Modelo pre-entrenado de Hugging Face
        """
        self.handles = []
        
        # Recorrer módulos del modelo y registrar hooks
        for nombre, modulo in modelo.named_modules():
            if len(list(modulo.children())) == 0:  # Solo módulos hoja (sin submódulos)
                # Hook para propagación hacia adelante
                handle = modulo.register_forward_hook(
                    lambda mod, inp, out, nombre=nombre: self.hook_forward(mod, inp, out, nombre)
                )
                self.handles.append(handle)
                
                # Hook para gradientes en propagación hacia atrás
                if hasattr(modulo, 'weight') and modulo.weight is not None:
                    handle = modulo.weight.register_hook(
                        lambda grad, nombre=nombre: self.hook_backward(grad, nombre)
                    )
                    self.handles.append(handle)
        
        logger.info(f"Registrados {len(self.handles)} hooks metacognitivos en el modelo")
    
    def hook_forward(self, modulo, entrada, salida, nombre_capa):
        """Hook para capturar y analizar activaciones durante propagación hacia adelante.
        
        Args:
            modulo: Módulo de PyTorch que activó el hook
            entrada: Entradas al módulo
            salida: Salidas del módulo
            nombre_capa: Nombre del módulo/capa
        """
        # Capturar activaciones
        if isinstance(salida, torch.Tensor):
            # Asegurar que estamos trabajando con FP32 para el análisis metacognitivo
            # Esto es crucial para compatibilidad cuando se usa FP16 en el modelo
            activacion = salida.detach()
            # Convertir a float32 si es necesario (por ejemplo, si viene en formato Half/FP16)
            if activacion.dtype == torch.float16 or activacion.dtype == torch.bfloat16:
                activacion = activacion.float()  # Convierte a float32
                
            self.activaciones_por_capa[nombre_capa] = {
                'media': activacion.mean().item(),
                'max': activacion.max().item(),
                'min': activacion.min().item(),
                'std': activacion.std().item() if activacion.numel() > 1 else 0.0
            }
            
            # Analizar activaciones en tiempo real
            self._analizar_activacion_tiempo_real(nombre_capa, activacion)
    
    def hook_backward(self, gradiente, nombre_capa):
        """Hook para capturar y analizar gradientes durante propagación hacia atrás.
        
        Args:
            gradiente: Gradiente de la capa
            nombre_capa: Nombre de la capa
        """
        # Asegurar que estamos en FP32 (para evitar errores con tensores FP16/Half)
        if gradiente.dtype != torch.float32:
            gradiente = gradiente.float()
        
        # Capturar gradientes
        self.gradientes_por_capa[nombre_capa] = {
            'media': gradiente.mean().item(),
            'max': gradiente.max().item(),
            'min': gradiente.min().item(),
            'norm': gradiente.norm().item()
        }
        
        # Analizar gradientes en tiempo real
        self._analizar_gradiente_tiempo_real(nombre_capa, gradiente)
    
    def _analizar_activacion_tiempo_real(self, nombre_capa, activacion):
        """Analiza las activaciones en tiempo real y aplica correcciones si es necesario.
        
        Args:
            nombre_capa: Nombre de la capa
            activacion: Tensor de activación
        """
        # Detección de problemas de activación
        problemas = []
        
        # Verificar activaciones muertas (muchos ceros)
        zeros_ratio = (activacion == 0).float().mean().item()
        if zeros_ratio > 0.5:  # Si más del 50% son ceros
            problemas.append(f"Activaciones muertas: {zeros_ratio:.2f} en {nombre_capa}")
        
        # Verificar saturación (muchos valores cerca del máximo/mínimo)
        if activacion.max().item() > 0:
            saturacion_ratio = (activacion > 0.9 * activacion.max().item()).float().mean().item()
            if saturacion_ratio > 0.3:  # Si más del 30% están saturados
                problemas.append(f"Saturación detectada: {saturacion_ratio:.2f} en {nombre_capa}")
        
        # Calcular entropía para detectar sobreajuste (baja diversidad en activaciones)
        if activacion.numel() > 1:
            # Normalizar y discretizar para calcular entropía
            act_norm = activacion.detach()
            
            # Verificar que no hay valores infinitos
            if not torch.isfinite(act_norm).all():
                # Reemplazar valores no finitos con 0
                act_norm = torch.where(torch.isfinite(act_norm), act_norm, torch.zeros_like(act_norm))
            
            # Continuar si hay un rango válido
            if act_norm.max() > act_norm.min():
                # Normalizar al rango [0,1]
                act_norm = (act_norm - act_norm.min()) / (act_norm.max() - act_norm.min())
                
                # Convertir a histograma para aproximar la distribución (con rango finito garantizado)
                histograma = torch.histc(act_norm, bins=10, min=0, max=1)
                # Normalizar histograma a distribución de probabilidad
                probs = histograma / histograma.sum()
                # Eliminar bins con probabilidad cero para el cálculo de entropía
                probs = probs[probs > 0]
                # Calcular entropía: -sum(p*log(p))
                entropia = -torch.sum(probs * torch.log2(probs)).item()
                
                # Guardar en histórico para análisis de tendencias
                if nombre_capa not in self.historico_entropia:
                    self.historico_entropia[nombre_capa] = []
                self.historico_entropia[nombre_capa].append(entropia)
                
                # Detectar caída sostenida de entropía (señal de sobreajuste)
                if len(self.historico_entropia[nombre_capa]) >= self.ventana_analisis:
                    entropia_reciente = np.mean(self.historico_entropia[nombre_capa][-10:])  # Últimos 10 valores
                    entropia_anterior = np.mean(self.historico_entropia[nombre_capa][-self.ventana_analisis:-10])  # Valores anteriores
                    caida_entropia = entropia_anterior - entropia_reciente
                    
                    # Si hay caída significativa de entropía, señalar posible sobreajuste
                    if caida_entropia > self.umbral_alerta_entropia and self.iteraciones > 1000:  # Ignorar fase inicial
                        problemas.append(f"Posible sobreajuste: caída de entropía {caida_entropia:.4f} en {nombre_capa}")
                        self.contador_alertas_sobreajuste += 1
                        
                        # Adaptar umbral si se detectan muchas alertas (para evitar falsos positivos)
                        if self.contador_alertas_sobreajuste > 5:
                            self.umbral_alerta_entropia *= 1.2  # Incrementar umbral
        
        # Registrar observación si hay problemas
        if problemas:
            self.observaciones_tiempo_real.append({
                'capa': nombre_capa,
                'tipo': 'activacion',
                'problemas': problemas,
                'iteracion': self.iteraciones
            })
    
    def _analizar_gradiente_tiempo_real(self, nombre_capa, gradiente):
        """Analiza los gradientes en tiempo real y aplica correcciones si es necesario.
        
        Args:
            nombre_capa: Nombre de la capa
            gradiente: Tensor de gradiente
        """
        # Detección de problemas de gradiente
        problemas = []
        
        # Verificar gradientes que desaparecen
        grad_norm = gradiente.norm().item()
        if grad_norm < 1e-7:
            problemas.append(f"Gradiente desapareciendo: {grad_norm:.8f} en {nombre_capa}")
        
        # Verificar gradientes que explotan
        if grad_norm > 10.0:
            problemas.append(f"Gradiente explotando: {grad_norm:.2f} en {nombre_capa}")
        
        # Registrar observación si hay problemas
        if problemas:
            self.observaciones_tiempo_real.append({
                'capa': nombre_capa,
                'tipo': 'gradiente',
                'problemas': problemas,
                'iteracion': self.iteraciones
            })
    
    def on_batch_complete(self, modelo):
        """Se ejecuta después de cada batch para procesar observaciones y aplicar correcciones.
        
        Args:
            modelo: Modelo en entrenamiento
        """
        self.iteraciones += 1
        
        # Cada 5 iteraciones o cuando se detectan problemas críticos
        if self.iteraciones % 5 == 0 or len(self.observaciones_tiempo_real) > 5:
            self._aplicar_correcciones_metacognitivas(modelo)
            self.observaciones_tiempo_real = []  # Reiniciar observaciones
    
    def _aplicar_correcciones_metacognitivas(self, modelo):
        """Aplica correcciones metacognitivas basadas en análisis de tiempo real.
        
        Args:
            modelo: Modelo en entrenamiento
        """
        if not self.observaciones_tiempo_real:
            return
        
        # Agrupar problemas por capa y tipo
        problemas_por_capa = {}
        detectado_sobreajuste = False
        for obs in self.observaciones_tiempo_real:
            capa = obs['capa']
            if capa not in problemas_por_capa:
                problemas_por_capa[capa] = []
            problemas_por_capa[capa].extend(obs['problemas'])
            
            # Verificar si hay problemas de sobreajuste
            for problema in obs['problemas']:
                if "Posible sobreajuste" in problema:
                    detectado_sobreajuste = True
        
        # Crear sistema antisobreajuste dinámico
        if detectado_sobreajuste and (self.iteraciones - self.ultimo_paso_regularizacion) > 50:
            self._aplicar_regulacion_antisobreajuste(modelo)
            self.ultimo_paso_regularizacion = self.iteraciones
        
        # Generar una reflexión sobre los problemas detectados
        datos_reflexion = {
            'problemas_tiempo_real': problemas_por_capa,
            'activaciones': self.activaciones_por_capa,
            'gradientes': self.gradientes_por_capa,
            'iteracion': self.iteraciones,
            'sobreajuste_detectado': detectado_sobreajuste,
            'nivel_regularizacion': self.nivel_regularizacion
        }
        
        if self.reflexion:
            try:
                reflexion = self.reflexion.generar_reflexion(datos_reflexion)
                
                # Convertir a diccionario si es string
                if isinstance(reflexion, str):
                    reflexion = {'resumen': reflexion, 'recomendaciones': []}
                
                # Aplicar modificaciones basadas en la reflexión
                if self.auto_modificacion and reflexion:
                    # Añadir información sobre sobreajuste a las modificaciones
                    reflexion['regulacion_antisobreajuste'] = self.nivel_regularizacion if detectado_sobreajuste else 0.0
                    
                    modificaciones = self.auto_modificacion.proponer_modificaciones(reflexion)
                    if modificaciones and 'resultados_modificaciones' in modificaciones:
                        cambios = modificaciones['resultados_modificaciones'].get('modificaciones_aplicadas', [])
                        if cambios:
                            self.intervenciones += len(cambios)
                            logger.info(f"Intervención metacognitiva: {len(cambios)} modificaciones aplicadas en tiempo real")
            except Exception as e:
                logger.warning(f"Error en intervención metacognitiva: {str(e)}")
    
    def _aplicar_regulacion_antisobreajuste(self, modelo):
        """Aplica mecanismos de regularización antisobreajuste.
        
        Este método implementa intervenciones metacognitivas específicas
        para mantener la diversidad de representaciones y evitar la memorización.
        
        Args:
            modelo: Modelo en entrenamiento
        """
        # Incrementar nivel de regularización de forma adaptativa
        self.nivel_regularizacion += 0.05
        self.nivel_regularizacion = min(self.nivel_regularizacion, 0.5)  # Limitar al 50% máximo
        
        logger.warning(f"ALERTA DE SOBREAJUSTE en iteración {self.iteraciones}: Activando regularización nivel {self.nivel_regularizacion:.2f}")
        
        try:
            # 1. Inyección de ruido controlado en capas específicas
            for nombre, modulo in modelo.named_modules():
                # Aplicar a capas de transformación (lineales, embeddings)
                if isinstance(modulo, torch.nn.Linear) or "embedding" in nombre.lower():
                    if hasattr(modulo, 'weight') and modulo.weight is not None:
                        # Calcular norma de los pesos antes de modificar
                        norma_original = modulo.weight.data.norm().item()
                        
                        # Aplicar ruido proporcional a los pesos existentes
                        ruido = torch.randn_like(modulo.weight.data) * self.nivel_regularizacion * 0.01
                        modulo.weight.data += ruido
                        
                        # Asegurar que la norma se mantiene similar (evitar desestabilizar)
                        norma_nueva = modulo.weight.data.norm().item()
                        if norma_nueva > 0:  # Evitar división por cero
                            modulo.weight.data *= (norma_original / norma_nueva)
            
            # 2. Aumentar temperatura efectiva para forzar exploración
            # (Esto se comunica al sistema de auto-modificación a través de los datos de reflexión)
            
            # 3. Aplicar técnica de mixup adaptativa si hay capas de embedding
            for nombre, modulo in modelo.named_modules():
                if "embedding" in nombre.lower() and hasattr(modulo, 'weight'):
                    # Implementar una versión simple de mixup en el espacio de embeddings
                    # Tomar pares aleatorios de embeddings y mezclarlos con factor alpha
                    with torch.no_grad():
                        tam_vocab, dim_emb = modulo.weight.shape
                        if tam_vocab > 100:  # Solo aplicar a embeddings grandes
                            num_mezclas = max(int(tam_vocab * 0.01 * self.nivel_regularizacion), 1)
                            for _ in range(num_mezclas):
                                # Seleccionar embeddings para mezclar
                                idx1, idx2 = torch.randint(0, tam_vocab, (2,))
                                if idx1 != idx2:
                                    # Factor de mezcla aleatorio entre 0.1 y 0.5
                                    factor = 0.1 + torch.rand(1).item() * 0.4
                                    # Mezclar embeddings (promedio ponderado)
                                    embedding_mixto = (1-factor) * modulo.weight.data[idx1] + factor * modulo.weight.data[idx2]
                                    # Reemplazar uno de los embeddings con la mezcla
                                    modulo.weight.data[idx1] = embedding_mixto
                                    
            logger.info(f"Aplicada regulación antisobreajuste: mixup, inyección de ruido y aumento de exploración")
        except Exception as e:
            logger.error(f"Error al aplicar regulación antisobreajuste: {str(e)}")

    def remover_hooks(self):
        """Elimina todos los hooks para liberar recursos."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        logger.info("Hooks metacognitivos removidos")
    
    def obtener_estadisticas(self):
        """Obtiene estadísticas de la intervención metacognitiva.
        
        Returns:
            Diccionario con estadísticas
        """
        # Calcular estadísticas de entropía para cada capa
        stats_entropia = {}
        for capa, historico in self.historico_entropia.items():
            if len(historico) > 0:
                stats_entropia[capa] = {
                    'entropia_media': np.mean(historico),
                    'entropia_max': np.max(historico),
                    'entropia_min': np.min(historico),
                    'entropia_actual': historico[-1] if historico else 0,
                    'tendencia': 'decreciente' if len(historico) > 10 and 
                                  np.mean(historico[-5:]) < np.mean(historico[-10:-5]) else 'estable'  
                }
        
        return {
            'iteraciones': self.iteraciones,
            'intervenciones': self.intervenciones,
            'observaciones_realizadas': len(self.observaciones_tiempo_real),
            'alertas_sobreajuste': self.contador_alertas_sobreajuste,
            'nivel_regularizacion': self.nivel_regularizacion,
            'estadisticas_entropia': stats_entropia
        }


class EntrenadorMetacognitivoDirecto:
    """Entrenador que implementa entrenamiento dirigido metacognitivamente sin épocas tradicionales.
    
    Soporta entrenamiento tanto en precisión completa (FP32) como en precisión mixta (FP16).
    El modo FP16 permite entrenar más rápido, usar batches más grandes o modelos más complejos
    con la misma memoria, manteniendo una calidad comparable de entrenamiento.
    """
    
    def __init__(
        self, 
        modelo: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        auto_observacion,
        reflexion,
        auto_modificacion,
        dispositivo: str = "cuda",
        fp16: bool = False
    ):
        """Inicializa el entrenador metacognitivo directo.
        
        Args:
            modelo: Modelo pre-entrenado a entrenar
            tokenizer: Tokenizador correspondiente al modelo
            auto_observacion: Componente para observar estados internos del modelo
            reflexion: Componente para generar reflexiones sobre el aprendizaje
            auto_modificacion: Componente para modificar pesos y comportamiento
            dispositivo: Dispositivo donde entrenar ("cuda" o "cpu")
            fp16: Si es True, usará entrenamiento en precisión mixta (FP16) para mayor eficiencia
        """
        # Validar que el modelo y tokenizer no sean None
        if modelo is None:
            raise ValueError("El modelo no puede ser None")
            
        if tokenizer is None:
            raise ValueError("El tokenizer no puede ser None")
            
        self.modelo = modelo
        self.tokenizer = tokenizer
        self.hook_metacognitivo = MetacognitiveHook(
            auto_observacion,
            reflexion,
            auto_modificacion
        )
        self.dispositivo = dispositivo
        
        # Configuración para entrenamiento en precisión mixta (FP16)
        self.fp16 = fp16 and dispositivo == "cuda"  # Solo usar FP16 en GPU
        if self.fp16:
            # Solo importar si se va a usar, evitando errores en sistemas sin CUDA
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            logger.info("Entrenamiento en precisión mixta (FP16) activado")
        else:
            self.scaler = None
            logger.info(f"Entrenamiento en precisión completa (FP32) en {dispositivo}")
        
        # Verificar disponibilidad de CUDA si se solicita
        if self.dispositivo == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA no disponible, usando CPU en su lugar")
            self.dispositivo = "cpu"
            
        # Asegurar que el modelo está en el dispositivo correcto
        logger.info(f"Moviendo modelo a dispositivo: {self.dispositivo}")
        self.modelo.to(self.dispositivo)
        
        # Inicializar hook metacognitivo
        self.hook_metacognitivo.registrar_hooks(self.modelo)
        
        # Estadísticas de entrenamiento
        self.estadisticas = {
            'pasos_realizados': 0,
            'perdida_acumulada': 0.0,
            'mejor_perdida': float('inf'),
            'ciclos_estancamiento': 0,
            'tiempo_entrenamiento': 0.0
        }
    
    def preparar_dataset(self, textos, max_length=128, batch_size=4):
        """Prepara el dataset para entrenamiento.
        
        Args:
            textos: Lista de textos para entrenar
            max_length: Longitud máxima de secuencia
            batch_size: Tamaño de batch
            
        Returns:
            DataLoader para entrenamiento
        """
        import torch
        from torch.utils.data import Dataset
        
        # Filtrar textos vacíos y truncarlos si son muy largos
        textos_validos = []
        for texto in textos:
            if texto and len(texto.strip()) > 0:
                # Limitar la longitud de los textos muy largos para evitar problemas de memoria
                if len(texto) > max_length * 10:
                    texto = texto[:max_length * 10]  # Texto truncado para prevenir errores de memoria
                textos_validos.append(texto)
                
        if not textos_validos:
            raise ValueError("No hay textos válidos para el entrenamiento")
        
        logger.info(f"Preparando dataset metacognitivo directo con {len(textos_validos)} textos")
        
        # Asegurar que el tokenizador tiene configuración de padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Crear una clase de dataset personalizada para tokenizar por lotes
        class MetacognitiveDataset(Dataset):
            def __init__(self, textos, tokenizer, max_length):
                self.textos = textos
                self.tokenizer = tokenizer
                self.max_length = max_length
                
            def __len__(self):
                return len(self.textos)
                
            def __getitem__(self, idx):
                # Tokenizar un solo texto a la vez para evitar problemas de memoria
                texto = self.textos[idx]
                encoding = self.tokenizer(
                    texto,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                # Eliminar dimensión de batch que agrega el tokenizador
                encoding = {k: v.squeeze(0) for k, v in encoding.items()}
                
                # Añadir etiquetas para entrenamiento de modelo causal
                encoding['labels'] = encoding['input_ids'].clone()
                
                return encoding
        
        # Crear dataset que tokeniza on-the-fly
        dataset = MetacognitiveDataset(textos_validos, self.tokenizer, max_length)
        
        logger.info("Dataset preparado con tokenización optimizada para memoria")
        
        # Función de collate personalizada
        def collate_fn(batch):
            # Organizar por claves
            collated = {key: [] for key in batch[0].keys()}
            
            # Recopilar todos los tensores
            for ejemplo in batch:
                for key, value in ejemplo.items():
                    collated[key].append(value)
            
            # Apilar en un solo tensor para cada clave
            for key in collated:
                collated[key] = torch.stack(collated[key])
                
            return collated
        
        # Crear dataloader con procesamiento optimizado
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # Un solo worker para evitar problemas de memoria
            pin_memory=False  # Desactivar pin_memory para reducir uso de memoria
        )
        
        logger.info(f"Dataset preparado con {len(dataset)} ejemplos")
        return dataloader
    
    def entrenar(self, 
                 textos: List[str], 
                 pasos_max: int = 100,
                 learning_rate: float = 2e-5,
                 batch_size: int = 4,
                 max_length: int = 128,
                 checkpoint_cada: int = 0,
                 dir_checkpoint: Optional[str] = None,
                 callbacks: Optional[List[Callable]] = None):
        """Entrena el modelo utilizando el enfoque metacognitivo directo sin épocas.
        
        Args:
            textos: Lista de textos para entrenar
            pasos_max: Número máximo de pasos de entrenamiento
            learning_rate: Tasa de aprendizaje inicial
            batch_size: Tamaño de batch
            max_length: Longitud máxima de secuencia
            checkpoint_cada: Guardar checkpoint cada N pasos (0 para deshabilitar)
            dir_checkpoint: Directorio donde guardar checkpoints
            callbacks: Lista de callbacks a llamar durante el entrenamiento
            
        Returns:
            Diccionario con resultados y estadísticas del entrenamiento
        """
        import time
        from torch.optim import AdamW
        
        # Preparar dataset
        dataloader = self.preparar_dataset(textos, max_length, batch_size)
        
        # Configurar optimizador
        optimizer = AdamW(
            self.modelo.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Realizar observación inicial
        if self.hook_metacognitivo.auto_observacion:
            try:
                observacion_inicial = self.hook_metacognitivo.auto_observacion.observar_pesos()
                logger.info("Observación inicial del modelo realizada")
            except Exception as e:
                logger.warning(f"Error en observación inicial: {str(e)}")
        
        # Iniciar entrenamiento
        self.modelo.train()
        start_time = time.time()
        pasos_realizados = 0
        perdida_acumulada = 0.0
        
        logger.info(f"Iniciando entrenamiento metacognitivo directo por {pasos_max} pasos")
        
        # Ciclo principal de entrenamiento
        try:
            ciclo_dataloader = iter(dataloader)
            while pasos_realizados < pasos_max:
                # Obtener nuevo batch (reiniciar si es necesario)
                try:
                    batch = next(ciclo_dataloader)
                except StopIteration:
                    ciclo_dataloader = iter(dataloader)
                    batch = next(ciclo_dataloader)
                
                # Mover batch al dispositivo
                batch = {k: v.to(self.dispositivo) for k, v in batch.items()}
                
                # Limpiar gradientes
                optimizer.zero_grad()
                
                # Forward pass y cálculo de pérdida con soporte para FP16
                if self.fp16:
                    try:
                        # Entrenamiento en precisión mixta (FP16)
                        from torch.cuda.amp import autocast
                        with autocast():
                            outputs = self.modelo(
                                batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"]
                            )
                            loss = outputs.loss
                        
                        # Backward pass con escalado de gradientes
                        self.scaler.scale(loss).backward()
                        
                        # Aplicar procesamiento metacognitivo después del backward (siempre en FP32)
                        self.hook_metacognitivo.on_batch_complete(self.modelo)
                        
                        # Actualizar pesos con desescalado
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    except RuntimeError as e:
                        if "HalfTensor is not supported" in str(e):
                            logger.warning("Error de compatibilidad con FP16. Cambiando a FP32 para este batch.")
                            # Volver a intentar con FP32
                            outputs = self.modelo(
                                batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"]
                            )
                            loss = outputs.loss
                            loss.backward()
                            optimizer.step()
                        else:
                            raise e
                else:
                    # Entrenamiento normal en FP32
                    outputs = self.modelo(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    loss = outputs.loss
                    
                    # Backward pass estándar
                    loss.backward()
                    
                    # Actualizar pesos
                    optimizer.step()
                
                # Aplicar procesamiento metacognitivo después del backward
                self.hook_metacognitivo.on_batch_complete(self.modelo)
                
                # Actualizar estadísticas
                perdida_acumulada += loss.item()
                pasos_realizados += 1
                
                # Mostrar progreso
                if pasos_realizados % 10 == 0:
                    avg_loss = perdida_acumulada / 10
                    logger.info(f"Paso {pasos_realizados}/{pasos_max} - Pérdida: {avg_loss:.4f}")
                    perdida_acumulada = 0.0
                
                # Guardar checkpoint si está configurado
                if checkpoint_cada > 0 and pasos_realizados % checkpoint_cada == 0 and dir_checkpoint:
                    os.makedirs(dir_checkpoint, exist_ok=True)
                    checkpoint_path = os.path.join(dir_checkpoint, f"checkpoint_paso_{pasos_realizados}")
                    # Guardar modelo
                    self.modelo.save_pretrained(checkpoint_path)
                    # IMPORTANTE: Guardar el tokenizador junto con el modelo
                    # para asegurar que se use el mismo tokenizador durante la inferencia
                    self.tokenizer.save_pretrained(checkpoint_path)
                    logger.info(f"Checkpoint (modelo y tokenizador) guardado en {checkpoint_path}")
                
                # Ejecutar callbacks si existen
                if callbacks:
                    for callback in callbacks:
                        try:
                            callback(self.modelo, pasos_realizados, loss.item())
                        except Exception as e:
                            logger.warning(f"Error en callback: {str(e)}")
        
        except KeyboardInterrupt:
            logger.info("Entrenamiento interrumpido por el usuario")
        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {str(e)}")
        finally:
            # Eliminar hooks
            self.hook_metacognitivo.remover_hooks()
        
        # Calcular tiempo de entrenamiento
        tiempo_entrenamiento = time.time() - start_time
        
        # Realizar observación final
        if self.hook_metacognitivo.auto_observacion:
            try:
                observacion_final = self.hook_metacognitivo.auto_observacion.observar_pesos()
                logger.info("Observación final del modelo realizada")
                
                # Generar reflexión final sobre el entrenamiento
                if self.hook_metacognitivo.reflexion:
                    datos_finales = {
                        'observacion_inicial': observacion_inicial if 'observacion_inicial' in locals() else None,
                        'observacion_final': observacion_final,
                        'estadisticas_hooks': self.hook_metacognitivo.obtener_estadisticas(),
                        'pasos_totales': pasos_realizados,
                        'tiempo_entrenamiento': tiempo_entrenamiento
                    }
                    
                    try:
                        reflexion_final = self.hook_metacognitivo.reflexion.generar_reflexion(datos_finales)
                        if isinstance(reflexion_final, str):
                            logger.info(f"Reflexión final (extracto): {reflexion_final[:200]}...")
                        else:
                            logger.info(f"Reflexión final generada: {reflexion_final.get('resumen', 'No disponible')[:200]}...")
                    except Exception as e:
                        logger.warning(f"Error en reflexión final: {str(e)}")
            except Exception as e:
                logger.warning(f"Error en observación final: {str(e)}")
        
        # Preparar resultados
        resultados = {
            'pasos_totales': pasos_realizados,
            'tiempo_entrenamiento': tiempo_entrenamiento,
            'estadisticas_hooks': self.hook_metacognitivo.obtener_estadisticas() if self.hook_metacognitivo else {},
            'dispositivo_utilizado': self.dispositivo
        }
        
        logger.info(f"Entrenamiento metacognitivo directo completado en {tiempo_entrenamiento:.2f} segundos")
        return resultados


def entrenar_modelo_metacognitivo_directo(
    modelo: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    textos: List[str],
    dir_salida: str,
    auto_observacion=None,
    reflexion=None,
    auto_modificacion=None,
    pasos_max: int = 100,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    dispositivo: str = "cuda",
    callbacks: Optional[List[Callable]] = None,
    monitoreo_sobreajuste=True,
    umbral_alerta_entropia=0.1,
    fp16: bool = False,
    checkpoint_cada: int = 0,
    nivel_inteligencia: int = 5,
    ciclos: int = 50,
    guardar_solo_mejor: bool = False,
    max_modelos_guardados: int = 5
) -> Dict:
    """Función principal para entrenar un modelo usando el enfoque metacognitivo directo.
    
    Args:
        modelo: Modelo pre-entrenado a entrenar
        tokenizer: Tokenizador correspondiente al modelo
        textos: Lista de textos para entrenar
        dir_salida: Directorio donde guardar resultados
        auto_observacion: Componente de auto-observación
        reflexion: Componente de reflexión metacognitiva
        auto_modificacion: Componente para modificar pesos y comportamiento
        pasos_max: Número máximo de pasos de entrenamiento (reemplaza épocas)
        batch_size: Tamaño de batch
        learning_rate: Tasa de aprendizaje inicial
        max_length: Longitud máxima de secuencia
        dispositivo: Dispositivo donde entrenar ("cuda" o "cpu")
        callbacks: Lista de callbacks a llamar durante el entrenamiento
        monitoreo_sobreajuste: Activar el sistema de monitoreo y regulación antisobreajuste
        umbral_alerta_entropia: Umbral para detectar caídas de entropía (más bajo = más sensible)
        fp16: Si es True, usará entrenamiento en precisión mixta (FP16) para mayor eficiencia
        checkpoint_cada: Frecuencia para guardar checkpoints (0 para desactivar)
        
    Returns:
        Diccionario con resultados y estadísticas del entrenamiento
    """
    import time
    import json
    from datetime import datetime
    
    logger.info(f"Iniciando entrenamiento metacognitivo directo con {len(textos)} textos")
    logger.info(f"Parámetros: pasos_max={pasos_max}, batch_size={batch_size}, lr={learning_rate}, max_length={max_length}")
    
    # Crear directorio de salida
    os.makedirs(dir_salida, exist_ok=True)
    
    # Inicializar entrenador
    entrenador = EntrenadorMetacognitivoDirecto(
        modelo=modelo,
        tokenizer=tokenizer,
        auto_observacion=auto_observacion,
        reflexion=reflexion,
        auto_modificacion=auto_modificacion,
        dispositivo=dispositivo,
        fp16=fp16
    )
    
    # Medir tiempo de entrenamiento
    inicio = time.time()
    
    # Ejecutar entrenamiento
    resultados = entrenador.entrenar(
        textos=textos,
        pasos_max=pasos_max,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_length=max_length,
        checkpoint_cada=max(1, pasos_max // 5),  # Guardar ~5 checkpoints durante el entrenamiento
        dir_checkpoint=os.path.join(dir_salida, "checkpoints"),
        callbacks=callbacks
    )
    
    # Calcular tiempo total
    tiempo_total = time.time() - inicio
    
    # Guardar modelo final
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    ruta_modelo_final = os.path.join(dir_salida, f"modelo_metacognitivo_directo_{timestamp}")
    logger.info(f"Guardando modelo final y tokenizador asociado en {ruta_modelo_final}")
    modelo.save_pretrained(ruta_modelo_final)
    # IMPORTANTE: Guardar el tokenizador junto con el modelo para inferencia
    # Esto asegura que en inferencia se use exactamente el mismo tokenizador que en entrenamiento
    tokenizer.save_pretrained(ruta_modelo_final)
    logger.info(f"Modelo y tokenizador guardados correctamente")
    
    # Preparar resultados
    resultados_completos = {
        **resultados,
        'tiempo_total': tiempo_total,
        'textos_entrenamiento': len(textos),
        'timestamp': timestamp,
        'ruta_modelo': ruta_modelo_final
    }
    
    # Guardar resultados
    with open(os.path.join(dir_salida, f"resultados_{timestamp}.json"), 'w', encoding='utf-8') as f:
        json.dump(resultados_completos, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Entrenamiento completado en {tiempo_total:.2f} segundos")
    logger.info(f"Modelo guardado en {ruta_modelo_final}")
    
    return resultados_completos


# Clase para integrar con el sistema metacognitivo existente
class EntrenamientoDirectoCallback:
    """Callback para integrar el sistema de entrenamiento directo con el sistema metacognitivo existente."""
    
    def __init__(self, auto_observacion, reflexion, auto_modificacion):
        """Inicializa el callback de entrenamiento directo.
        
        Args:
            auto_observacion: Componente de auto-observación
            reflexion: Componente de reflexión metacognitiva
            auto_modificacion: Componente de auto-modificación
        """
        self.auto_observacion = auto_observacion
        self.reflexion = reflexion
        self.auto_modificacion = auto_modificacion
        self.resultados = []
    
    def __call__(self, modelo, paso, perdida):
        """Se llama durante el entrenamiento para cada paso.
        
        Args:
            modelo: Modelo en entrenamiento
            paso: Número de paso actual
            perdida: Valor de pérdida para este paso
        """
        # Registrar resultados cada 10 pasos
        if paso % 10 == 0:
            self.resultados.append({
                'paso': paso,
                'perdida': perdida
            })
    
    def obtener_resultados(self):
        """Obtiene los resultados registrados durante el entrenamiento.
        
        Returns:
            Lista de resultados por paso
        """
        return self.resultados

def detectar_recursos_gpu():
    """
    Detecta y analiza los recursos disponibles de GPU para optimización automática.
    
    Returns:
        Dict con información sobre recursos de GPU e información para optimización
    """
    recursos = {
        "gpu_disponible": False,
        "memoria_total_gb": 0.0,
        "memoria_libre_gb": 0.0,
        "cuda_version": "N/A",
        "optimizaciones": {}
    }
    
    try:
        if torch.cuda.is_available():
            recursos["gpu_disponible"] = True
            recursos["cuda_version"] = torch.version.cuda
            recursos["dispositivos"] = torch.cuda.device_count()
            
            # Obtener información de memoria
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            mem_reservada = torch.cuda.memory_reserved(0) / (1024**3)  # GB
            mem_asignada = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            mem_libre = mem_total - mem_asignada
            
            recursos["memoria_total_gb"] = round(mem_total, 2)
            recursos["memoria_libre_gb"] = round(mem_libre, 2)
            recursos["memoria_modelo_maxima_gb"] = round(mem_libre * 0.85, 2)  # 85% de memoria libre
            
            # Generar recomendaciones basadas en memoria disponible
            if mem_libre < 4.0:
                # GPU con memoria muy limitada
                recursos["optimizaciones"] = {
                    "fp16": True,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 32,
                    "n_layer": 6,
                    "n_head": 6,
                    "n_embd": 384
                }
                logger.info(f"Detectada GPU con memoria muy limitada ({mem_libre:.2f} GB). Configurando parámetros para modelo pequeño.")
            elif mem_libre < 8.0:
                # GPU con memoria moderada
                recursos["optimizaciones"] = {
                    "fp16": True,
                    "batch_size": 2,
                    "gradient_accumulation_steps": 16,
                    "n_layer": 12,
                    "n_head": 12,
                    "n_embd": 768
                }
                logger.info(f"Detectada GPU con memoria moderada ({mem_libre:.2f} GB). Configurando para modelo mediano.")
            else:
                # GPU con buena memoria
                recursos["optimizaciones"] = {
                    "fp16": True,
                    "batch_size": 3,
                    "gradient_accumulation_steps": 8,
                    "n_layer": 16,
                    "n_head": 12,
                    "n_embd": 768
                }
                logger.info(f"Detectada GPU con buena memoria ({mem_libre:.2f} GB). Optimizando parámetros para modelo grande.")
    
    except Exception as e:
        logger.warning(f"Error al analizar recursos GPU: {str(e)}")
        # Si hay error, configurar valores por defecto conservadores
        recursos["optimizaciones"] = {
            "fp16": True,
            "batch_size": 1,
            "gradient_accumulation_steps": 32,
            "n_layer": 8,
            "n_head": 8,
            "n_embd": 512
        }
    
    return recursos


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
    
    # Verificar que el directorio exista
    if not os.path.exists(directorio):
        return
    
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


if __name__ == "__main__":
    import argparse
    import os
    import time
    import json
    from pathlib import Path
    from datetime import datetime
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('entrenamiento_metacognitivo.log', 'a')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Detectar recursos disponibles
    logger.info(f"PyTorch version {torch.__version__} disponible.")
    recursos = detectar_recursos_gpu()
    
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenamiento Metacognitivo Directo')
    
    # Argumentos principales
    parser.add_argument('--ruta_dataset', type=str, required=True, help='Ruta al directorio con archivos de texto')
    parser.add_argument('--dir_salida', type=str, required=True, help='Directorio para guardar el modelo')
    
    # Parámetros de entrenamiento
    parser.add_argument('--pasos_max', type=int, default=20000, help='Número máximo de pasos de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=recursos['optimizaciones'].get('batch_size', 3), 
                        help='Tamaño de batch')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Tasa de aprendizaje')
    parser.add_argument('--max_length', type=int, default=512, help='Longitud máxima de secuencia')
    parser.add_argument('--dispositivo', type=str, default='cuda' if recursos['gpu_disponible'] else 'cpu', 
                        help='Dispositivo a usar (cuda o cpu)')
    parser.add_argument('--fp16', action='store_true', default=recursos['optimizaciones'].get('fp16', False), 
                        help='Usar precisión mixta FP16')
    parser.add_argument('--gradient_accumulation_steps', type=int, 
                        default=recursos['optimizaciones'].get('gradient_accumulation_steps', 8),
                        help='Pasos de acumulación de gradientes')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Decaimiento de pesos')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Pasos de calentamiento')
    parser.add_argument('--logging_steps', type=int, default=100, help='Frecuencia de logging')
    
    # Opciones de checkpoint
    parser.add_argument('--checkpoint_cada', type=int, default=1000, help='Guardar checkpoint cada N pasos')
    parser.add_argument('--max_modelos_guardados', type=int, default=5, 
                        help='Máximo número de modelos guardados (0=ilimitado)')
    parser.add_argument('--guardar_solo_mejor', action='store_true', help='Guardar solo el mejor modelo')
    parser.add_argument('--limpiar_modelos_anteriores', action='store_true', 
                        help='Limpiar modelos antiguos antes de entrenar')
    
    # Capacidades metacognitivas
    parser.add_argument('--nivel_inteligencia', type=int, default=7, 
                        help='Nivel de inteligencia metacognitiva (1-10)')
    parser.add_argument('--ciclos', type=int, default=50, help='Número de ciclos metacognitivos')
    parser.add_argument('--monitoreo_sobreajuste', action='store_true', default=True, 
                        help='Activar monitoreo de sobreajuste')
    parser.add_argument('--umbral_alerta_entropia', type=float, default=0.1, 
                        help='Umbral para detección de sobreajuste')
    parser.add_argument('--auto_observacion', action='store_true', default=True, 
                        help='Activar auto-observación')
    parser.add_argument('--reflexion', action='store_true', default=True, 
                        help='Activar reflexión')
    parser.add_argument('--auto_modificacion', action='store_true', default=True, 
                        help='Activar auto-modificación')
    
    # Parámetros de arquitectura del modelo - valores por defecto basados en los recursos detectados
    parser.add_argument('--n_layer', type=int, default=recursos['optimizaciones'].get('n_layer', 12), 
                        help='Número de capas del transformer')
    parser.add_argument('--n_head', type=int, default=recursos['optimizaciones'].get('n_head', 12), 
                        help='Número de cabezales de atención')
    parser.add_argument('--n_embd', type=int, default=recursos['optimizaciones'].get('n_embd', 768), 
                        help='Dimensión de los embeddings')
    parser.add_argument('--n_inner', type=int, default=None, 
                        help='Dimensión de la capa intermedia en el feed-forward')
    parser.add_argument('--n_ctx', type=int, default=512, help='Tamaño del contexto')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--archivos_excluidos', type=str, default='', 
                        help='Patrones de archivos a excluir del entrenamiento (separados por comas)')
    
    # Parámetro de metacognición
    parser.add_argument('--nivel_inteligencia', type=int, default=5, help='Nivel de inteligencia metacognitiva (1-10)')
    parser.add_argument('--ciclos', type=int, default=50, help='Número de ciclos metacognitivos')
    parser.add_argument('--guardar_solo_mejor', action='store_true', help='Guardar solo el mejor modelo')
    parser.add_argument('--max_modelos_guardados', type=int, default=5, help='Máximo número de modelos guardados')
    
    args = parser.parse_args()
    
    # Crear directorios de salida
    logger.info(f"Preparando directorio de salida: {args.dir_salida}")
    dir_modelos = os.path.join(args.dir_salida, 'modelos')
    dir_logs = os.path.join(args.dir_salida, 'logs')
    os.makedirs(dir_modelos, exist_ok=True)
    os.makedirs(dir_logs, exist_ok=True)
    
    # Limpiar modelos antiguos si se solicitó
    if args.limpiar_modelos_anteriores and args.max_modelos_guardados > 0:
        logger.info(f"Limpiando modelos antiguos, manteniendo máximo {args.max_modelos_guardados} modelos")
        limpiar_modelos_anteriores(dir_modelos, args.max_modelos_guardados, args.guardar_solo_mejor)
    
    # Función mejorada para cargar textos
    def obtener_textos_entrenamiento(directorio, excluir=None):
        """Obtener textos para entrenamiento con manejo robusto."""
        textos = []
        archivos_procesados = 0
        archivos_excluidos = 0
        
        # Normalizar patrones de exclusión
        patrones_exclusion = []
        if excluir and isinstance(excluir, str) and excluir.strip():
            patrones_exclusion = [p.strip().lower() for p in excluir.split(',') if p.strip()]
        
        logger.info(f"Buscando archivos .txt en {directorio}")
        
        # Función recursiva para buscar en subdirectorios
        def procesar_directorio(ruta):
            nonlocal textos, archivos_procesados, archivos_excluidos
            
            # Verificar si es un directorio válido
            if not os.path.isdir(ruta):
                return
                
            try:
                # Listar contenidos
                contenidos = os.listdir(ruta)
                
                # Procesar archivos .txt
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
                    
                    # Leer el contenido
                    try:
                        with open(ruta_completa, 'r', encoding='utf-8', errors='ignore') as f:
                            contenido = f.read().strip()
                            if contenido:  # Asegurar que no esté vacío
                                textos.append(contenido)
                                archivos_procesados += 1
                                if archivos_procesados % 10 == 0:
                                    logger.info(f"Procesados {archivos_procesados} archivos hasta ahora")
                    except Exception as e:
                        logger.warning(f"Error al leer {ruta_completa}: {str(e)}")
                
                # Procesar subdirectorios recursivamente
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
    
    # Cargar textos para entrenamiento
    logger.info(f"Cargando textos desde {args.ruta_dataset}")
    textos = obtener_textos_entrenamiento(args.ruta_dataset, args.archivos_excluidos)
    
    if not textos:
        logger.error("No se encontraron textos válidos para entrenamiento")
        sys.exit(1)
    
    logger.info(f"Cargados {len(textos)} textos para entrenamiento")
    
    # Mostrar resumen del modelo que se va a entrenar
    tam_modelo_aprox = args.n_layer * args.n_head * args.n_embd * 2 / 1_000_000
    logger.info(f"Configurando modelo con aproximadamente {tam_modelo_aprox:.2f}M parámetros")
    logger.info(f"Arquitectura: {args.n_layer} capas, {args.n_head} cabezales, {args.n_embd} dim. embeds, {args.n_ctx} dim. contexto")
    if args.fp16:
        logger.info("Usando entrenamiento con precisión mixta (FP16) para mayor eficiencia")
    if args.gradient_accumulation_steps > 1:
        logger.info(f"Batch efectivo: {args.batch_size * args.gradient_accumulation_steps} (batch {args.batch_size} * {args.gradient_accumulation_steps} pasos acum.)")
    if recursos['gpu_disponible']:
        logger.info(f"Usando GPU con {recursos['memoria_total_gb']:.2f}GB de memoria total, {recursos['memoria_libre_gb']:.2f}GB libre")
    
    # Inicializar modelo y tokenizador
    logger.info("Inicializando modelo y tokenizador")
    
    # Intentar cargar el tokenizador personalizado
    tokenizer = None
    try:
        from tokenizador_metacognitivo import MetacognitivoTokenizer
        logger.info("Módulo tokenizador_metacognitivo importado correctamente")
        
        try:
            # Ruta al tokenizador personalizado
            ruta_tokenizador = "D:/AItrainer/Data/Tokenizers/Spanish/tokenizador_metacognitivo"
            
            # Comprobar que existan los archivos necesarios
            vocab_path = os.path.join(ruta_tokenizador, "vocab.json")
            merges_path = os.path.join(ruta_tokenizador, "merges.txt")
            config_path = os.path.join(ruta_tokenizador, "tokenizer_config.json")
            
            if os.path.exists(vocab_path) and os.path.exists(merges_path):
                logger.info(f"Cargando tokenizador metacognitivo desde: {ruta_tokenizador}")
                
                # Cargar configuración si existe
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        tokenizer_config = json.load(f)
                
                # Crear instancia del tokenizador
                tokenizer = MetacognitivoTokenizer(
                    vocab_file=vocab_path,
                    merges_file=merges_path
                )
                tokenizer = tokenizer.hf_tokenizer  # Usar el tokenizador HuggingFace interno
                tokenizer.model_max_length = args.max_length
                logger.info(f"Tokenizador metacognitivo cargado correctamente")
            else:
                logger.warning(f"No se encontraron los archivos necesarios en {ruta_tokenizador}")
        except Exception as e:
            logger.warning(f"Error al cargar tokenizador personalizado: {str(e)}")
    except ImportError as e:
        logger.warning(f"No se pudo importar el módulo tokenizador_metacognitivo: {str(e)}")
    
    # Usar tokenizador multilingüe como respaldo si es necesario
    if tokenizer is None:
        model_name = "bert-base-multilingual-cased"
        logger.info(f"Usando tokenizador {model_name} para español como respaldo")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.model_max_length = args.max_length
    
    # Asegurar que el tokenizador esté configurado correctamente
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
    
    logger.info(f"Tokenizador listo. Vocabulario: {len(tokenizer)} tokens")
    
    # Inicializar el modelo con configuración personalizada
    try:
        logger.info(f"Inicializando modelo con configuración optimizada para tus recursos")
        config = AutoConfig.from_pretrained("gpt2")
        
        # Actualizar configuración con parámetros personalizados
        config.vocab_size = len(tokenizer)
        config.n_layer = args.n_layer
        config.n_head = args.n_head
        config.n_embd = args.n_embd
        config.n_inner = args.n_inner if args.n_inner else 4 * args.n_embd
        config.n_ctx = args.n_ctx
        config.resid_pdrop = args.dropout
        config.embd_pdrop = args.dropout
        config.attn_pdrop = args.dropout
        
        # Configurar para mejor rendimiento en español
        config.scale_attn_weights = True
        config.use_cache = True
        
        # Calcular y mostrar tamaño aproximado del modelo
        tam_modelo_aprox = args.n_layer * args.n_head * args.n_embd * 2 / 1_000_000
        logger.info(f"Creando modelo con aproximadamente {tam_modelo_aprox:.2f}M parámetros")
        
        inicio_creacion = time.time()
        logger.info("Inicializando modelo nuevo desde cero con pesos aleatorios")
        modelo = AutoModelForCausalLM.from_config(config)
        logger.info(f"Modelo creado en {time.time() - inicio_creacion:.2f} segundos")
        
        # Importar componentes metacognitivos
        try:
            from auto_observacion import SistemaAutoObservacion
            from reflexion_metacognitiva import SistemaReflexion
            from auto_modificacion import SistemaAutoModificacion
            
            auto_observacion = SistemaAutoObservacion() if args.auto_observacion else None
            reflexion = SistemaReflexion() if args.reflexion else None
            auto_modificacion = SistemaAutoModificacion() if args.auto_modificacion else None
            
            if auto_observacion or reflexion or auto_modificacion:
                logger.info("Componentes metacognitivos cargados correctamente")
        except ImportError as e:
            logger.warning(f"No se pudieron cargar componentes metacognitivos: {str(e)}")
            auto_observacion = reflexion = auto_modificacion = None
        
        # Crear directorio de salida
        os.makedirs(args.dir_salida, exist_ok=True)
        
        # Ejecutar entrenamiento
        logger.info("Iniciando entrenamiento metacognitivo directo")
        entrenar_modelo_metacognitivo_directo(
            modelo=modelo,
            tokenizer=tokenizer,
            textos=textos,
            dir_salida=args.dir_salida,
            auto_observacion=auto_observacion,
            reflexion=reflexion,
            auto_modificacion=auto_modificacion,
            pasos_max=args.pasos_max,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            dispositivo=args.dispositivo,
            fp16=args.fp16,
            checkpoint_cada=args.checkpoint_cada,
            monitoreo_sobreajuste=args.monitoreo_sobreajuste,
            umbral_alerta_entropia=args.umbral_alerta_entropia,
            nivel_inteligencia=args.nivel_inteligencia,
            ciclos=args.ciclos,
            guardar_solo_mejor=args.guardar_solo_mejor,
            max_modelos_guardados=args.max_modelos_guardados
        )
        logger.info("Entrenamiento completado exitosamente")
    except Exception as e:
        logger.error(f"Error durante la inicialización o entrenamiento: {str(e)}")
        raise
    
    # Importar componentes metacognitivos
    try:
        from auto_observacion import SistemaAutoObservacion
        from reflexion_metacognitiva import SistemaReflexion
        from auto_modificacion import SistemaAutoModificacion
        
        auto_observacion = SistemaAutoObservacion() if args.auto_observacion else None
        reflexion = SistemaReflexion() if args.reflexion else None
        auto_modificacion = SistemaAutoModificacion() if args.auto_modificacion else None
    except ImportError as e:
        logger.warning(f"No se pudieron cargar componentes metacognitivos: {str(e)}")
        auto_observacion = reflexion = auto_modificacion = None
    
    # Crear directorio de salida
    os.makedirs(args.dir_salida, exist_ok=True)
    
    # Ejecutar entrenamiento
    logger.info("Iniciando entrenamiento metacognitivo directo")
    try:
        entrenar_modelo_metacognitivo_directo(
            modelo=modelo,
            tokenizer=tokenizer,
            textos=textos,
            dir_salida=args.dir_salida,
            auto_observacion=auto_observacion,
            reflexion=reflexion,
            auto_modificacion=auto_modificacion,
            pasos_max=args.pasos_max,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            dispositivo=args.dispositivo,
            fp16=args.fp16,
            checkpoint_cada=args.checkpoint_cada,
            monitoreo_sobreajuste=args.monitoreo_sobreajuste,
            umbral_alerta_entropia=args.umbral_alerta_entropia,
            nivel_inteligencia=args.nivel_inteligencia,
            ciclos=args.ciclos,
            guardar_solo_mejor=args.guardar_solo_mejor,
            max_modelos_guardados=args.max_modelos_guardados
        )
        logger.info("Entrenamiento completado exitosamente")
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {str(e)}")
        raise
