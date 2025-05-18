#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo de Auto-Observación para el Sistema Metacognitivo Avanzado.

Este módulo permite que el modelo acceda a representaciones de sus propios
pesos, parámetros y gradientes durante el entrenamiento, facilitando
la introspección y metacognición.
"""

import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from transformers import PreTrainedModel, AutoModelForCausalLM

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class AutoObservacion:
    """
    Clase que implementa la capacidad de auto-observación para modelos de lenguaje.
    Permite que el modelo acceda a representaciones de sus propios pesos y gradientes.
    """
    
    def __init__(self, modelo: PreTrainedModel, config: Dict = None):
        """
        Inicializa el módulo de auto-observación.
        
        Args:
            modelo: Modelo pre-entrenado a observar
            config: Configuración del módulo de auto-observación
        """
        self.modelo = modelo
        self.config = config or {}
        self.device = next(modelo.parameters()).device
        
        # Configuración por defecto
        self.config.setdefault('max_params_per_layer', 20)   # Número máximo de parámetros a mostrar por capa (reducido)
        self.config.setdefault('gradient_tracking', False)   # Desactivar seguimiento de gradientes por defecto
        self.config.setdefault('activation_tracking', False) # Activar seguimiento de activaciones
        self.config.setdefault('histogram_bins', 5)          # Bins para histogramas de distribución (reducido)
        self.config.setdefault('max_layers_to_track', 10)    # Límite de capas a analizar en detalle
        
        # Registro de observaciones
        self.observaciones = {
            'pesos': {},
            'gradientes': {},
            'activaciones': {},
            'estadisticas': {},
            'historia': []
        }
        
        # Registrar hooks para capturar gradientes si está activado
        self.hooks = []
        if self.config['gradient_tracking']:
            self._registrar_hooks_gradientes()
        
        logger.info(f"Módulo de auto-observación inicializado para modelo {type(modelo).__name__}")
    
    def _registrar_hooks_gradientes(self):
        """Registra hooks para capturar gradientes durante el backpropagation."""
        for nombre, param in self.modelo.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, nombre=nombre: self._capturar_gradiente(nombre, grad)
                )
                self.hooks.append(hook)
        
        logger.info(f"Registrados {len(self.hooks)} hooks para captura de gradientes")
    
    def _capturar_gradiente(self, nombre: str, gradiente: torch.Tensor):
        """
        Captura y almacena el gradiente de un parámetro específico.
        
        Args:
            nombre: Nombre del parámetro
            gradiente: Tensor de gradiente
        """
        # Almacenar solo una representación resumida del gradiente
        self.observaciones['gradientes'][nombre] = {
            'mean': gradiente.mean().item(),
            'std': gradiente.std().item(),
            'min': gradiente.min().item(),
            'max': gradiente.max().item(),
            'norm': gradiente.norm().item(),
            'histogram': self._calcular_histograma(gradiente)
        }
    
    def _calcular_histograma(self, tensor: torch.Tensor) -> Dict[str, float]:
        """
        Calcula un histograma de valores para un tensor.
        
        Args:
            tensor: Tensor para calcular histograma
            
        Returns:
            Diccionario con bins y frecuencias
        """
        # Asegurar que el tensor está en CPU y convertir a float32 si es necesario
        # (torch.histc no funciona con tensores de tipo Half/float16)
        tensor_flat = tensor.detach().cpu().float().flatten()
        
        try:
            # Calcular mínimo y máximo para los bins
            tensor_min = tensor_flat.min().item()
            tensor_max = tensor_flat.max().item()
            
            # Si min==max, ajustar para evitar errores
            if tensor_min == tensor_max:
                tensor_min -= 1e-5
                tensor_max += 1e-5
            
            # Calcular histograma
            hist = torch.histc(
                tensor_flat, 
                bins=self.config['histogram_bins'],
                min=tensor_min,
                max=tensor_max
            )
        except RuntimeError as e:
            logger.warning(f"Error al calcular histograma: {str(e)}, usando método alternativo")
            # Método alternativo usando numpy si torch.histc falla
            tensor_np = tensor_flat.numpy()
            hist_np, edges_np = np.histogram(tensor_np, bins=self.config['histogram_bins'])
            hist = torch.tensor(hist_np)
            edges = torch.tensor(edges_np)
            return {f"{edges[i]:.4f}_{edges[i+1]:.4f}": hist[i].item() for i in range(len(hist))}
        
        # Continuar con el cálculo normal si no hubo excepciones
        edges = torch.linspace(
            tensor_flat.min().item(),
            tensor_flat.max().item(),
            self.config['histogram_bins'] + 1
        )
        
        # Crear diccionario de histograma
        histograma = {}
        for i in range(self.config['histogram_bins']):
            bin_key = f"{edges[i]:.4f}_{edges[i+1]:.4f}"
            histograma[bin_key] = hist[i].item()
        
        return histograma
    
    def observar_pesos(self) -> Dict[str, Any]:
        """
        Captura y analiza los pesos actuales del modelo.
        
        Returns:
            Diccionario con información sobre los pesos
        """
        resultados = {}
        
        # Obtener lista de parámetros y limitar a un número máximo de capas
        parametros = [(nombre, param) for nombre, param in self.modelo.named_parameters() if param.requires_grad]
        
        # Seleccionar capas importantes (primeras, últimas y algunas intermedias)
        max_layers = self.config['max_layers_to_track']
        if len(parametros) > max_layers:
            # Tomar algunas capas del principio, medio y final
            inicio = max_layers // 3
            final = max_layers // 3
            medio = max_layers - inicio - final
            
            seleccionados = parametros[:inicio]  # Primeras capas
            
            # Capas intermedias espaciadas uniformemente
            if medio > 0 and len(parametros) > inicio + final:
                paso = (len(parametros) - inicio - final) // (medio + 1)
                for i in range(medio):
                    idx = inicio + (i + 1) * paso
                    if idx < len(parametros) - final:
                        seleccionados.append(parametros[idx])
            
            # Últimas capas
            seleccionados.extend(parametros[-final:])  
            
            parametros = seleccionados
            logger.info(f"Limitando análisis a {len(parametros)} capas de {len(self.modelo.state_dict())} totales")
        
        # Analizar solo las capas seleccionadas
        for nombre, param in parametros:
            # Usar CPU para reducir uso de memoria GPU
            tensor = param.detach().cpu()
            
            # Información básica (ligera)
            resultados[nombre] = {
                'shape': list(tensor.shape),
                'mean': tensor.mean().item(),
                'std': tensor.std().item(),
                'min': tensor.min().item(),
                'max': tensor.max().item(),
                'norm': tensor.norm().item()
            }
            
            # Histograma solo para tensores no muy grandes
            if tensor.numel() < 100000:  # Limitar a tensores no muy grandes
                resultados[nombre]['histogram'] = self._calcular_histograma(tensor)
            
            # Muestra muy reducida de valores
            if tensor.numel() > self.config['max_params_per_layer']:
                indices = torch.randperm(tensor.numel())[:self.config['max_params_per_layer']]
                muestra = tensor.flatten()[indices].tolist()
            else:
                # Limitar aún más si es necesario
                muestra = tensor.flatten()[:self.config['max_params_per_layer']].tolist()
            
            resultados[nombre]['muestra'] = muestra
        
        # Actualizar observaciones
        self.observaciones['pesos'] = resultados
        return resultados
    
    def generar_resumen_capa(self, nombre_capa: str) -> str:
        """
        Genera un resumen textual de una capa específica.
        
        Args:
            nombre_capa: Nombre de la capa a resumir
            
        Returns:
            Texto descriptivo de la capa
        """
        if nombre_capa not in self.observaciones['pesos']:
            return f"No hay información disponible para la capa {nombre_capa}"
        
        info = self.observaciones['pesos'][nombre_capa]
        
        resumen = [
            f"Resumen de la capa: {nombre_capa}",
            f"Forma: {info['shape']}",
            f"Parámetros totales: {np.prod(info['shape']):,}",
            f"Estadísticas: Media={info['mean']:.6f}, Desv={info['std']:.6f}",
            f"Rango: [{info['min']:.6f}, {info['max']:.6f}]",
            f"Norma: {info['norm']:.6f}"
        ]
        
        # Añadir información de gradientes si está disponible
        if nombre_capa in self.observaciones['gradientes']:
            grad_info = self.observaciones['gradientes'][nombre_capa]
            resumen.extend([
                "",
                "Información de gradientes:",
                f"Media={grad_info['mean']:.6f}, Desv={grad_info['std']:.6f}",
                f"Rango: [{grad_info['min']:.6f}, {grad_info['max']:.6f}]",
                f"Norma: {grad_info['norm']:.6f}"
            ])
        
        return "\n".join(resumen)
    
    def generar_representacion_textual(self) -> str:
        """
        Genera una representación textual completa del estado del modelo.
        
        Returns:
            Texto descriptivo del modelo
        """
        if not self.observaciones['pesos']:
            self.observar_pesos()
        
        # Generar resumen general
        num_capas = len(self.observaciones['pesos'])
        num_params = sum(np.prod(info['shape']) for info in self.observaciones['pesos'].values())
        
        lineas = [
            "REPRESENTACIÓN INTERNA DEL MODELO",
            "=" * 40,
            f"Tipo de modelo: {type(self.modelo).__name__}",
            f"Número de capas: {num_capas}",
            f"Parámetros totales: {num_params:,}",
            "=" * 40,
            ""
        ]
        
        # Añadir resúmenes por grupos de capas
        grupos_capas = self._agrupar_capas()
        for grupo, capas in grupos_capas.items():
            lineas.append(f"\n## Grupo: {grupo}")
            lineas.append("-" * 40)
            
            for capa in capas[:5]:  # Limitar a 5 capas por grupo para no sobrecargar
                lineas.append(self.generar_resumen_capa(capa))
                lineas.append("-" * 30)
            
            if len(capas) > 5:
                lineas.append(f"... y {len(capas) - 5} capas más en este grupo")
        
        return "\n".join(lineas)
    
    def _agrupar_capas(self) -> Dict[str, List[str]]:
        """
        Agrupa las capas del modelo por categorías para facilitar su análisis.
        
        Returns:
            Diccionario con grupos de capas
        """
        grupos = {}
        
        for nombre in self.observaciones['pesos'].keys():
            # Identificar grupo basado en el nombre
            if 'embed' in nombre:
                grupo = 'Embeddings'
            elif 'attention' in nombre:
                grupo = 'Attention'
            elif 'layer' in nombre and 'norm' in nombre:
                grupo = 'LayerNorm'
            elif 'mlp' in nombre or 'ffn' in nombre:
                grupo = 'FeedForward'
            elif 'head' in nombre:
                grupo = 'Heads'
            else:
                grupo = 'Otros'
            
            if grupo not in grupos:
                grupos[grupo] = []
            
            grupos[grupo].append(nombre)
        
        return grupos
    
    def guardar_observaciones(self, ruta_directorio: str):
        """
        Guarda las observaciones actuales en archivos para análisis posterior.
        
        Args:
            ruta_directorio: Directorio donde guardar las observaciones
        """
        os.makedirs(ruta_directorio, exist_ok=True)
        
        # Guardar representación textual
        with open(os.path.join(ruta_directorio, "representacion_modelo.txt"), "w", encoding="utf-8") as f:
            f.write(self.generar_representacion_textual())
        
        # Guardar estadísticas en formato JSON para análisis posterior
        import json
        
        # Convertir tensores a tipos serializables
        def convertir_para_json(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(os.path.join(ruta_directorio, "estadisticas_modelo.json"), "w", encoding="utf-8") as f:
            # Filtrar y convertir datos para JSON
            datos_json = {
                'pesos': {k: {k2: convertir_para_json(v2) for k2, v2 in v.items() if k2 != 'muestra'} 
                         for k, v in self.observaciones['pesos'].items()},
                'gradientes': self.observaciones['gradientes']
            }
            json.dump(datos_json, f, indent=2)
        
        logger.info(f"Observaciones guardadas en {ruta_directorio}")
    
    def liberar_recursos(self):
        """Libera los hooks y recursos utilizados por el módulo."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.observaciones = {
            'pesos': {},
            'gradientes': {},
            'activaciones': {},
            'estadisticas': {},
            'historia': []
        }
        logger.info("Recursos del módulo de auto-observación liberados")


# Función auxiliar para crear una instancia del módulo
def crear_modulo_auto_observacion(modelo, config=None):
    """
    Crea y configura un módulo de auto-observación para un modelo.
    
    Args:
        modelo: Modelo pre-entrenado a observar
        config: Configuración opcional
        
    Returns:
        Instancia de AutoObservacion
    """
    return AutoObservacion(modelo, config)


if __name__ == "__main__":
    # Ejemplo de uso
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Cargar un modelo pequeño para demostración
    modelo_nombre = "gpt2"  # Usar un modelo pequeño para pruebas
    modelo = AutoModelForCausalLM.from_pretrained(modelo_nombre)
    
    # Crear módulo de auto-observación
    auto_obs = crear_modulo_auto_observacion(modelo)
    
    # Observar pesos
    auto_obs.observar_pesos()
    
    # Generar y mostrar representación textual
    representacion = auto_obs.generar_representacion_textual()
    print(representacion)
    
    # Guardar observaciones
    auto_obs.guardar_observaciones("./observaciones_demo")
    
    # Liberar recursos
    auto_obs.liberar_recursos()
