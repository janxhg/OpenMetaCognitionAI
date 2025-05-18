#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para entrenamiento metacognitivo de modelos

Este módulo proporciona funciones para entrenar modelos con auto-reflexión y
adaptación de hiperparámetros durante el proceso.
"""

import os
import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from transformers import (
    PreTrainedTokenizer, 
    PreTrainedModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

logger = logging.getLogger(__name__)

class TextoDataset(Dataset):
    """Dataset personalizado para entrenamiento de modelos de lenguaje."""
    
    def __init__(self, textos: List[str], tokenizer: PreTrainedTokenizer, max_length: int = 256):
        """
        Inicializa el dataset con textos y un tokenizador.
        
        Args:
            textos: Lista de textos para entrenar
            tokenizer: Tokenizador a utilizar
            max_length: Longitud máxima de secuencia
        """
        # Asegurar que tenemos textos válidos
        textos_filtrados = [texto for texto in textos if texto and len(texto.strip()) > 0]
        
        if not textos_filtrados:
            raise ValueError("No hay textos válidos para el entrenamiento")
            
        logger.info(f"Creando dataset con {len(textos_filtrados)} textos válidos")
        
        # Asegurar que el tokenizador tenga configuración de padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Configurando pad_token = eos_token para dataset")
        
        # Tokenizar de manera eficiente con batchs más pequeños si hay muchos textos
        if len(textos_filtrados) > 100:
            # Tokenizar en batches para evitar problemas de memoria
            batch_size = 50
            all_encodings = []
            
            for i in range(0, len(textos_filtrados), batch_size):
                batch_texts = textos_filtrados[i:i+batch_size]
                batch_encodings = tokenizer(
                    batch_texts,
                    truncation=True, 
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                    return_attention_mask=True
                )
                all_encodings.append(batch_encodings)
            
            # Combinar resultados
            self.encodings = {}
            for key in all_encodings[0].keys():
                self.encodings[key] = torch.cat([enc[key] for enc in all_encodings])
        else:
            # Tokenización directa para conjuntos pequeños
            self.encodings = tokenizer(
                textos_filtrados, 
                truncation=True, 
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
                return_attention_mask=True
            )
        
        logger.info(f"Dataset creado con {len(self.encodings.input_ids)} ejemplos")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Obtiene un ítem del dataset."""
        try:
            item = {key: val[idx].clone() for key, val in self.encodings.items()}
            # Para causal language modeling, inputs = labels
            item["labels"] = item["input_ids"].clone()
            return item
        except Exception as e:
            logger.error(f"Error al acceder al índice {idx}: {str(e)}")
            # Devolver un ejemplo vacío con la estructura correcta como alternativa
            first_item = {key: val[0].clone() for key, val in self.encodings.items()}
            # Llenar con tokens de padding
            for k in first_item:
                first_item[k].fill_(0)  # Usar 0 como valor seguro
            return first_item
    
    def __len__(self) -> int:
        """Devuelve la longitud del dataset."""
        return len(self.encodings.input_ids)


def entrenar_modelo(
    modelo: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    textos: List[str],
    dir_salida: str,
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    estrategia_memoria: str = "gradiente_acumulado",
    max_length: int = 256,
    id_ciclo: Optional[int] = None,
    callbacks: Optional[List[callable]] = None,
    dispositivo: str = "cuda"
) -> Dict:
    """
    Entrena un modelo utilizando los textos proporcionados con capacidad metacognitiva.
    
    Args:
        modelo: Modelo pre-entrenado a fine-tunear
        tokenizer: Tokenizador correspondiente al modelo
        textos: Lista de textos para entrenamiento
        dir_salida: Directorio donde guardar resultados
        num_epochs: Número de epochs de entrenamiento
        batch_size: Tamaño del batch (reducir si hay problemas de memoria)
        learning_rate: Tasa de aprendizaje
        estrategia_memoria: Estrategia para manejar memoria limitada
        max_length: Longitud máxima de secuencia
        id_ciclo: Identificador del ciclo de entrenamiento
        callbacks: Lista de funciones de callback para introspección durante el entrenamiento
        dispositivo: Dispositivo de entrenamiento (cuda, cpu)
        
    Returns:
        Diccionario con métricas y resultados del entrenamiento
    """
    # Preparar directorio para este ciclo
    id_suffix = f"_ciclo{id_ciclo}" if id_ciclo is not None else ""
    output_dir = os.path.join(dir_salida, f"modelo_entrenado{id_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Ajustar configuración en función de memoria disponible
    if estrategia_memoria == "gradiente_acumulado":
        # Usar batches más pequeños pero acumular gradientes
        gradient_accumulation_steps = max(1, 8 // batch_size)
        logger.info(f"Usando acumulación de gradientes: {gradient_accumulation_steps} pasos")
    else:
        gradient_accumulation_steps = 1
    
    # Limitar cantidad de textos si es necesario
    if len(textos) > 50:
        logger.warning(f"Limitando a 50 textos para entrenamiento (de {len(textos)} disponibles)")
        textos = textos[:50]
    
    # Preparar dataset y data collator
    logger.info(f"Preparando dataset con {len(textos)} textos para entrenamiento")
    dataset = TextoDataset(textos, tokenizer, max_length=max_length)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # No usamos masked language modeling, sino causal LM para GPT
    )
    
    # Configuración de entrenamiento optimizada para memoria limitada
    fp16_enabled = torch.cuda.is_available() and dispositivo == "cuda"
    
    # Detectar si estamos en una GPU con memoria limitada
    memoria_limitada = False
    if torch.cuda.is_available() and dispositivo == "cuda":
        # Obtener memoria total disponible en GPU
        try:
            memoria_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memoria_limitada = memoria_total_gb < 8  # Considerar GPU con menos de 8GB como limitada
            logger.info(f"Memoria GPU total: {memoria_total_gb:.2f} GB")
            if memoria_limitada:
                logger.warning(f"GPU con memoria limitada detectada ({memoria_total_gb:.2f} GB). Aplicando optimizaciones de memoria.")
        except Exception as e:
            logger.warning(f"No se pudo determinar la memoria GPU: {str(e)}")
    
    # Ajustes especiales para GPUs con poca memoria
    if memoria_limitada:
        # Reducir parámetros más agresivamente
        batch_size = min(2, batch_size)  # Batch size muy pequeño
        gradient_accumulation_steps = max(8, gradient_accumulation_steps)  # Más acumulación
        max_length = min(max_length, 64)  # Secuencias más cortas
        logger.info(f"Ajustes para memoria limitada: batch_size={batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}, max_length={max_length}")
    
    # Configurar argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=fp16_enabled,  # Usar precisión media solo si tenemos CUDA
        fp16_opt_level="O1",  # Nivel de optimización para fp16 (O1 es un buen balance)
        dataloader_num_workers=0,  # Evitar problemas de memoria con dataloader
        disable_tqdm=False,
        report_to="none",  # No usar wandb u otros servicios externos
        # Opciones para ahorrar memoria adicional
        ddp_find_unused_parameters=False,
        optim="adamw_torch",  # Usar optimizer de PyTorch que es más eficiente en memoria
    )
    
    # Activar gradient checkpointing si es posible para ahorrar memoria (enorme diferencia en uso de VRAM)
    try:
        if hasattr(modelo, 'gradient_checkpointing_enable') and dispositivo == "cuda":
            modelo.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing activado para ahorrar memoria")
            
            # En modelos grandes, esto puede reducir el uso de memoria hasta un 70%
            if hasattr(modelo.config, 'use_cache'):
                # Desactivar past_key_values también ahorra memoria
                modelo.config.use_cache = False
                logger.info("Cache de KV desactivado para ahorrar memoria adicional")
    except Exception as e:
        logger.warning(f"No se pudo activar gradient checkpointing: {str(e)}")
        
    # Crear trainer con opciones avanzadas de memoria
    trainer = Trainer(
        model=modelo,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    # Implementar callbacks para introspección durante entrenamiento
    if callbacks:
        # Ejecutar callbacks pre-entrenamiento
        for callback in callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(trainer)
    
    # Verificar que el modelo y tokenizador estén correctamente configurados
    try:
        # Verificar modelo
        logger.info(f"Verificando modelo antes del entrenamiento ({type(modelo).__name__})")
        if not hasattr(modelo, 'config'):
            raise ValueError("El modelo no tiene configuración válida")
            
        # Verificar tokenizador
        logger.info(f"Verificando tokenizador ({type(tokenizer).__name__})")
        if tokenizer.pad_token is None:
            logger.warning("Tokenizador sin pad_token configurado. Configurando pad_token = eos_token")
            tokenizer.pad_token = tokenizer.eos_token
            
        # Garantizar que se han procesado textos válidos
        if not len(textos):
            raise ValueError("No hay textos para entrenar")
            
        logger.info(f"Configuración de entrenamiento: {batch_size=}, {num_epochs=}, {max_length=}, {learning_rate=}")
        
        # Iniciar entrenamiento con diagnóstico detallado
        logger.info(f"Iniciando entrenamiento del modelo con {len(textos)} textos")
        torch.cuda.empty_cache()  # Limpiar memoria GPU antes de entrenar
        
        # Ejecutar entrenamiento con seguimiento de memoria
        if torch.cuda.is_available():
            memoria_inicial = torch.cuda.memory_allocated() / (1024 * 1024)
            logger.info(f"Memoria GPU inicial: {memoria_inicial:.2f} MB")
        
        # Entrenamiento con manejo de errores específicos
        try:
            train_result = trainer.train()
            metrics = train_result.metrics
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                logger.error(f"Error de memoria CUDA. Intente reducir batch_size o max_length: {str(e)}")
                return {"estado": "error", "error": "CUDA OOM", "mensaje": str(e), 
                        "recomendacion": "Reduzca batch_size o max_length"}
            else:
                raise e
            
        # Guardar modelo y métricas
        logger.info(f"Entrenamiento completado. Guardando modelo en {output_dir}")
        trainer.save_model(output_dir)
        trainer.save_metrics("train", metrics)
        
        # Calcular pérdida promedio para introspección
        avg_loss = metrics.get("train_loss", 0)
        
        # Ejecutar callbacks post-entrenamiento
        if callbacks:
            for callback in callbacks:
                if hasattr(callback, 'on_train_end'):
                    try:
                        callback.on_train_end(trainer, avg_loss)
                    except Exception as callback_err:
                        logger.warning(f"Error en callback post-entrenamiento: {str(callback_err)}")
        
        # Información final del entrenamiento
        if torch.cuda.is_available():
            memoria_final = torch.cuda.memory_allocated() / (1024 * 1024)
            logger.info(f"Memoria GPU final: {memoria_final:.2f} MB")
            
        logger.info(f"Entrenamiento completado exitosamente. Pérdida final: {avg_loss:.4f}")
        
        return {
            "estado": "completado",
            "output_dir": output_dir,
            "metrics": metrics,
            "avg_loss": avg_loss,
            "tokens_procesados": len(dataset) * max_length * num_epochs
        }
        
    except Exception as e:
        import traceback
        error_detalle = traceback.format_exc()
        logger.error(f"Error durante el entrenamiento: {str(e)}\n{error_detalle}")
        return {
            "estado": "error",
            "error": str(e),
            "traceback": error_detalle,
            "fase": "entrenamiento"
        }


class IntrospectionCallback:
    """Callback para realizar introspección durante el entrenamiento."""
    
    def __init__(self, auto_observacion, reflexion, auto_modificacion):
        """
        Inicializa el callback con componentes introspectivos.
        
        Args:
            auto_observacion: Componente de auto-observación
            reflexion: Componente de reflexión metacognitiva
            auto_modificacion: Componente de auto-modificación
        """
        self.auto_observacion = auto_observacion
        self.reflexion = reflexion
        self.auto_modificacion = auto_modificacion
        self.observaciones = []
        
    def on_train_begin(self, trainer):
        """Se ejecuta al inicio del entrenamiento."""
        # Realizar observación inicial del modelo
        if self.auto_observacion:
            try:
                observacion_inicial = self.auto_observacion.observar_pesos()
                self.observaciones.append({
                    "momento": "inicio",
                    "observacion": observacion_inicial
                })
                logger.info("Observación inicial del modelo realizada")
            except Exception as e:
                logger.warning(f"Error en observación inicial: {str(e)}")
    
    def on_train_end(self, trainer, avg_loss):
        """Se ejecuta al final del entrenamiento."""
        # Realizar observación final del modelo
        if self.auto_observacion:
            try:
                observacion_final = self.auto_observacion.observar_pesos()
                self.observaciones.append({
                    "momento": "final",
                    "observacion": observacion_final
                })
                
                # Comparar estados para reflexión
                if len(self.observaciones) >= 2 and self.reflexion:
                    # Combinar todos los datos en un solo diccionario como espera la función
                    datos_combinados = {
                        "observacion_inicial": self.observaciones[0]["observacion"],
                        "observacion_final": self.observaciones[-1]["observacion"],
                        "perdida_promedio": avg_loss
                    }
                    
                    # Llamada corregida con un solo argumento
                    reflexion = self.reflexion.generar_reflexion(datos_combinados)
                    
                    # Verificar el tipo de retorno de generar_reflexion
                    if isinstance(reflexion, dict):
                        logger.info(f"Reflexión generada: {reflexion.get('resumen', 'No disponible')}")
                        
                        # Proponer modificaciones basadas en reflexión
                        if self.auto_modificacion:
                            modificaciones = self.auto_modificacion.proponer_modificaciones(reflexion)
                    elif isinstance(reflexion, str):
                        # Si es un string, adaptémoslo a un formato de diccionario
                        logger.info(f"Reflexión generada (formato texto): {reflexion[:100]}...")
                        
                        reflexion_dict = {
                            'resumen': reflexion,
                            'observaciones': {},
                            'recomendaciones': []
                        }
                        
                        # Proponer modificaciones basadas en reflexión convertida
                        if self.auto_modificacion:
                            modificaciones = self.auto_modificacion.proponer_modificaciones(reflexion_dict)
                        logger.info(f"Propuestas {len(modificaciones)} modificaciones")
                        
                        # En un sistema completo, aquí se aplicarían las modificaciones
                        # para el siguiente ciclo de entrenamiento
                
            except Exception as e:
                logger.warning(f"Error en introspección final: {str(e)}")
