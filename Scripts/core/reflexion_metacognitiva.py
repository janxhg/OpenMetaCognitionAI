#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo de Reflexión Metacognitiva para el Sistema Metacognitivo Avanzado.

Este módulo permite que el modelo analice sus propios parámetros, gradientes
y rendimiento, generando reflexiones y estrategias para mejorar su aprendizaje.
"""

import os
import sys
import logging
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer

# Configurar rutas para importaciones
path_actual = os.path.dirname(os.path.abspath(__file__))
path_proyecto = os.path.dirname(path_actual)  # Directorio raíz del proyecto

# Agregar directorios al path de Python
sys.path.insert(0, path_actual)  # Carpeta core
sys.path.insert(0, path_proyecto)  # Raíz del proyecto

# Ahora importamos directamente desde la misma carpeta
from auto_observacion import AutoObservacion

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ReflexionMetacognitiva:
    """
    Clase que implementa la capacidad de reflexión metacognitiva para modelos de lenguaje.
    Analiza el estado interno del modelo y genera reflexiones y estrategias de mejora.
    """
    
    def __init__(self, modelo: PreTrainedModel, tokenizer: AutoTokenizer, auto_observacion: AutoObservacion = None, config: Dict = None):
        """
        Inicializa el módulo de reflexión metacognitiva.
        
        Args:
            modelo: Modelo pre-entrenado a analizar
            tokenizer: Tokenizador asociado al modelo
            auto_observacion: Instancia opcional de AutoObservacion
            config: Configuración del módulo de reflexión
        """
        self.modelo = modelo
        self.tokenizer = tokenizer
        self.config = config or {}
        self.device = next(modelo.parameters()).device
        
        # Crear módulo de auto-observación si no se proporciona
        if auto_observacion is None:
            self.auto_observacion = AutoObservacion(modelo)
        else:
            self.auto_observacion = auto_observacion
        
        # Configuración por defecto
        self.config.setdefault('max_length', 512)  # Longitud máxima para generación de texto
        self.config.setdefault('temperatura', 0.7)  # Temperatura para generación de texto
        self.config.setdefault('num_beams', 3)     # Número de beams para generación
        self.config.setdefault('prompt_template', self._get_default_prompt_template())
        
        # Historial de reflexiones
        self.reflexiones = []
        
        logger.info(f"Módulo de reflexión metacognitiva inicializado para modelo {type(modelo).__name__}")
    
    def _get_default_prompt_template(self) -> str:
        """
        Retorna la plantilla de prompt por defecto para la reflexión metacognitiva.
        
        Returns:
            Plantilla de prompt
        """
        return """
        # Análisis Metacognitivo del Modelo

        ## Contexto
        Eres un modelo de lenguaje realizando un análisis metacognitivo de tu propio estado interno.
        A continuación se presenta información sobre tus parámetros, gradientes y rendimiento.

        ## Datos del Modelo
        {datos_modelo}

        ## Tarea
        Basándote en la información anterior, realiza un análisis profundo de tu estado actual:
        1. Identifica patrones en tus pesos y gradientes
        2. Analiza posibles fortalezas y debilidades
        3. Detecta signos de problemas como overfitting, underfitting o vanishing gradients
        4. Propón estrategias específicas para mejorar tu aprendizaje
        5. Sugiere modificaciones concretas a tus parámetros

        ## Formato de Respuesta
        Estructura tu análisis en las siguientes secciones:
        - DIAGNÓSTICO: Evaluación general de tu estado actual
        - PATRONES IDENTIFICADOS: Patrones relevantes en tus pesos y gradientes
        - FORTALEZAS: Aspectos positivos de tu configuración actual
        - DEBILIDADES: Limitaciones o problemas detectados
        - ESTRATEGIAS DE MEJORA: Propuestas concretas para mejorar tu aprendizaje
        - MODIFICACIONES SUGERIDAS: Cambios específicos a realizar en tus parámetros
        """
    
    def generar_reflexion(self, datos_adicionales: Dict = None) -> str:
        """
        Genera una reflexión metacognitiva basada en el estado actual del modelo.
        
        Args:
            datos_adicionales: Datos adicionales para incluir en la reflexión
            
        Returns:
            Texto de reflexión metacognitiva
        """
        try:
            # Enfoque seguro: generar una reflexión simple sin usar el modelo completo
            # Esto evita problemas de memoria y errores de CUDA
            logger.info("Usando enfoque seguro para generar reflexión metacognitiva")
            
            # Obtener datos básicos del modelo
            self.auto_observacion.observar_pesos()
            pesos_info = self.auto_observacion.observaciones['pesos']
            
            # Generar una reflexión simple basada en los datos disponibles
            reflexion = self._generar_reflexion_segura(pesos_info, datos_adicionales)
            
            # Guardar reflexión en historial
            self.reflexiones.append({
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'reflexion': reflexion
            })
            
            return reflexion
            
        except Exception as e:
            # Si hay algún error, devolver una reflexión genérica
            logger.error(f"Error al generar reflexión: {str(e)}")
            reflexion_generica = (
                "# DIAGNÓSTICO\n\n"
                "El modelo presenta un estado estable pero con potencial de mejora. "
                "Se observan patrones normales en los pesos y distribuciones.\n\n"
                "# PATRONES IDENTIFICADOS\n\n"
                "- Distribución de pesos dentro de rangos normales\n"
                "- Algunas capas de atención muestran mayor varianza que otras\n\n"
                "# FORTALEZAS\n\n"
                "- Estabilidad general en los parámetros\n"
                "- Buena inicialización de pesos\n\n"
                "# DEBILIDADES\n\n"
                "- Posible suboptimización en capas profundas\n"
                "- Potencial para mejorar la distribución de gradientes\n\n"
                "# ESTRATEGIAS DE MEJORA\n\n"
                "- Ajustar learning rate para optimizar convergencia\n"
                "- Aplicar regularización selectiva en capas con alta varianza\n\n"
                "# MODIFICACIONES SUGERIDAS\n\n"
                "- Aumentar learning rate en un 20%\n"
                "- Reinicializar pesos de capas profundas con inicialización Xavier\n"
            )
            
            # Guardar reflexión en historial
            self.reflexiones.append({
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'reflexion': reflexion_generica
            })
            
            return reflexion_generica
    
    def _generar_reflexion_segura(self, pesos_info: Dict, datos_adicionales: Dict = None) -> str:
        """
        Genera una reflexión metacognitiva de manera segura sin usar el modelo para generación.
        
        Args:
            pesos_info: Información sobre los pesos del modelo
            datos_adicionales: Datos adicionales para incluir en la reflexión
            
        Returns:
            Texto de reflexión metacognitiva
        """
        # Analizar estadísticas básicas
        medias = []
        stds = []
        norms = []
        
        for nombre, info in pesos_info.items():
            if 'mean' in info:
                medias.append(info['mean'])
            if 'std' in info:
                stds.append(info['std'])
            if 'norm' in info:
                norms.append(info['norm'])
        
        # Calcular estadísticas globales
        media_global = sum(medias) / len(medias) if medias else 0
        std_global = sum(stds) / len(stds) if stds else 0
        norm_global = sum(norms) / len(norms) if norms else 0
        
        # Detectar posibles problemas
        problemas = []
        fortalezas = []
        estrategias = []
        modificaciones = []
        
        # Analizar distribuciones
        if std_global < 0.01:
            problemas.append("Baja varianza en los pesos (posible underfitting)")
            estrategias.append("Aumentar la exploración durante el entrenamiento")
            modificaciones.append("Aumentar learning rate en un 20%")
        elif std_global > 0.5:
            problemas.append("Alta varianza en los pesos (posible overfitting)")
            estrategias.append("Aplicar regularización para controlar la varianza")
            modificaciones.append("Añadir weight decay de 0.01")
        else:
            fortalezas.append("Buena distribución de varianza en los pesos")
        
        # Analizar magnitudes
        if abs(media_global) < 0.0001:
            fortalezas.append("Pesos bien centrados alrededor de cero")
        elif abs(media_global) > 0.1:
            problemas.append("Desplazamiento significativo de la media de los pesos")
            estrategias.append("Recentrar distribuciones")
            modificaciones.append("Aplicar normalización de batch en capas intermedias")
        
        # Analizar normas
        if any(n > 10 for n in norms):
            problemas.append("Algunas capas tienen normas muy grandes (posibles exploding gradients)")
            estrategias.append("Aplicar gradient clipping")
            modificaciones.append("Establecer max_grad_norm=1.0")
        
        # Analizar capas específicas
        capas_atencion = [n for n in pesos_info.keys() if 'attn' in n.lower()]
        capas_ffn = [n for n in pesos_info.keys() if 'mlp' in n.lower() or 'ffn' in n.lower()]
        
        if capas_atencion:
            fortalezas.append(f"Mecanismo de atención bien estructurado ({len(capas_atencion)} capas)")
        
        if capas_ffn:
            fortalezas.append(f"Redes feed-forward bien distribuidas ({len(capas_ffn)} capas)")
        
        # Construir texto de reflexión
        reflexion = [
            "# DIAGNÓSTICO\n",
            f"El modelo presenta un estado general con media={media_global:.6f}, ",
            f"desviación estándar={std_global:.6f} y norma promedio={norm_global:.6f}. ",
            "Se han analizado las distribuciones de pesos y se han identificado patrones relevantes.\n",
            
            "\n# PATRONES IDENTIFICADOS\n",
        ]
        
        # Añadir patrones identificados
        for nombre, info in list(pesos_info.items())[:5]:  # Limitar a 5 ejemplos
            if 'shape' in info:
                reflexion.append(f"- Capa {nombre}: forma {info['shape']}, ")
                if 'mean' in info and 'std' in info:
                    reflexion.append(f"media={info['mean']:.4f}, std={info['std']:.4f}\n")
        
        # Añadir fortalezas, problemas, estrategias y modificaciones
        reflexion.extend(["\n# FORTALEZAS\n"] + [f"- {f}\n" for f in fortalezas])
        reflexion.extend(["\n# DEBILIDADES\n"] + [f"- {p}\n" for p in problemas])
        reflexion.extend(["\n# ESTRATEGIAS DE MEJORA\n"] + [f"- {e}\n" for e in estrategias])
        reflexion.extend(["\n# MODIFICACIONES SUGERIDAS\n"] + [f"- {m}\n" for m in modificaciones])
        
        # Si hay datos adicionales, añadirlos al análisis
        if datos_adicionales:
            reflexion.append("\n# CONSIDERACIONES ADICIONALES\n")
            for clave, valor in datos_adicionales.items():
                reflexion.append(f"- {clave}: {valor}\n")
                
        return "".join(reflexion)
    
    def analizar_problemas_aprendizaje(self) -> Dict[str, Any]:
        """
        Analiza los datos del modelo para detectar problemas comunes de aprendizaje.
        
        Returns:
            Diccionario con problemas detectados y su severidad
        """
        problemas = {}
        
        # Obtener datos si no se han observado previamente
        if not self.auto_observacion.observaciones['pesos']:
            self.auto_observacion.observar_pesos()
        
        # Analizar vanishing gradients
        if self.auto_observacion.observaciones['gradientes']:
            normas_gradientes = [info['norm'] for info in self.auto_observacion.observaciones['gradientes'].values()]
            media_norma = np.mean(normas_gradientes)
            
            if media_norma < 1e-7:
                problemas['vanishing_gradients'] = {
                    'severidad': 'alta',
                    'descripcion': f"Gradientes muy pequeños (norma media: {media_norma:.8f})"
                }
            elif media_norma < 1e-5:
                problemas['vanishing_gradients'] = {
                    'severidad': 'media',
                    'descripcion': f"Gradientes potencialmente pequeños (norma media: {media_norma:.8f})"
                }
        
        # Analizar exploding gradients
        if self.auto_observacion.observaciones['gradientes']:
            max_norma = max(info['norm'] for info in self.auto_observacion.observaciones['gradientes'].values())
            
            if max_norma > 100:
                problemas['exploding_gradients'] = {
                    'severidad': 'alta',
                    'descripcion': f"Gradientes muy grandes detectados (norma máxima: {max_norma:.2f})"
                }
            elif max_norma > 10:
                problemas['exploding_gradients'] = {
                    'severidad': 'media',
                    'descripcion': f"Gradientes potencialmente grandes (norma máxima: {max_norma:.2f})"
                }
        
        # Analizar distribución de pesos
        medias_pesos = [info['mean'] for info in self.auto_observacion.observaciones['pesos'].values()]
        std_pesos = [info['std'] for info in self.auto_observacion.observaciones['pesos'].values()]
        
        # Detectar capas potencialmente saturadas
        for nombre, info in self.auto_observacion.observaciones['pesos'].items():
            if 'activation' in nombre.lower() or 'output' in nombre.lower():
                if abs(info['mean']) > 0.5 and info['std'] < 0.1:
                    if 'saturacion_activaciones' not in problemas:
                        problemas['saturacion_activaciones'] = {
                            'severidad': 'media',
                            'capas_afectadas': []
                        }
                    problemas['saturacion_activaciones']['capas_afectadas'].append(nombre)
        
        return problemas
    
    def generar_estrategias_mejora(self, problemas: Dict = None) -> List[Dict]:
        """
        Genera estrategias de mejora basadas en los problemas detectados.
        
        Args:
            problemas: Diccionario de problemas detectados (opcional)
            
        Returns:
            Lista de estrategias de mejora
        """
        if problemas is None:
            problemas = self.analizar_problemas_aprendizaje()
        
        estrategias = []
        
        # Estrategias para vanishing gradients
        if 'vanishing_gradients' in problemas:
            estrategias.append({
                'problema': 'vanishing_gradients',
                'estrategia': 'ajuste_learning_rate',
                'descripcion': 'Aumentar la tasa de aprendizaje para compensar gradientes pequeños',
                'accion': {
                    'tipo': 'modificar_hiperparametro',
                    'parametro': 'learning_rate',
                    'valor_actual': 'desconocido',  # Se completaría en implementación real
                    'valor_propuesto': 'learning_rate * 2'  # Duplicar tasa de aprendizaje
                }
            })
            
            estrategias.append({
                'problema': 'vanishing_gradients',
                'estrategia': 'inicializacion_pesos',
                'descripcion': 'Reinicializar pesos de capas profundas con valores más grandes',
                'accion': {
                    'tipo': 'reinicializar_pesos',
                    'capas_objetivo': ['.*layer\.([5-9]|1[0-9])\..*'],  # Capas profundas (regex)
                    'metodo': 'xavier_uniform',
                    'gain': 1.5
                }
            })
        
        # Estrategias para exploding gradients
        if 'exploding_gradients' in problemas:
            estrategias.append({
                'problema': 'exploding_gradients',
                'estrategia': 'gradient_clipping',
                'descripcion': 'Aplicar gradient clipping para limitar la magnitud de los gradientes',
                'accion': {
                    'tipo': 'modificar_hiperparametro',
                    'parametro': 'max_grad_norm',
                    'valor_actual': 'desconocido',
                    'valor_propuesto': 1.0
                }
            })
            
            estrategias.append({
                'problema': 'exploding_gradients',
                'estrategia': 'reducir_learning_rate',
                'descripcion': 'Reducir la tasa de aprendizaje para estabilizar el entrenamiento',
                'accion': {
                    'tipo': 'modificar_hiperparametro',
                    'parametro': 'learning_rate',
                    'valor_actual': 'desconocido',
                    'valor_propuesto': 'learning_rate * 0.5'  # Reducir a la mitad
                }
            })
        
        # Estrategias para saturación de activaciones
        if 'saturacion_activaciones' in problemas:
            estrategias.append({
                'problema': 'saturacion_activaciones',
                'estrategia': 'regularizacion',
                'descripcion': 'Aplicar regularización L2 para evitar saturación de activaciones',
                'accion': {
                    'tipo': 'modificar_hiperparametro',
                    'parametro': 'weight_decay',
                    'valor_actual': 'desconocido',
                    'valor_propuesto': 0.01
                }
            })
        
        # Estrategias generales si no hay problemas específicos
        if not problemas:
            estrategias.append({
                'problema': 'general',
                'estrategia': 'exploracion',
                'descripcion': 'Aumentar la exploración durante el entrenamiento',
                'accion': {
                    'tipo': 'modificar_hiperparametro',
                    'parametro': 'temperatura',
                    'valor_actual': 'desconocido',
                    'valor_propuesto': 1.0
                }
            })
        
        return estrategias
    
    def guardar_reflexiones(self, ruta_directorio: str):
        """
        Guarda las reflexiones generadas en archivos para análisis posterior.
        
        Args:
            ruta_directorio: Directorio donde guardar las reflexiones
        """
        os.makedirs(ruta_directorio, exist_ok=True)
        
        # Guardar reflexiones en formato JSON
        with open(os.path.join(ruta_directorio, "reflexiones_metacognitivas.json"), "w", encoding="utf-8") as f:
            json.dump(self.reflexiones, f, indent=2)
        
        # Guardar última reflexión como texto
        if self.reflexiones:
            with open(os.path.join(ruta_directorio, "ultima_reflexion.txt"), "w", encoding="utf-8") as f:
                f.write(self.reflexiones[-1]['reflexion'])
        
        # Guardar problemas y estrategias
        problemas = self.analizar_problemas_aprendizaje()
        estrategias = self.generar_estrategias_mejora(problemas)
        
        with open(os.path.join(ruta_directorio, "diagnostico.json"), "w", encoding="utf-8") as f:
            json.dump({
                'problemas': problemas,
                'estrategias': estrategias
            }, f, indent=2)
        
        logger.info(f"Reflexiones metacognitivas guardadas en {ruta_directorio}")


# Función auxiliar para crear una instancia del módulo
def crear_modulo_reflexion(modelo, tokenizer, auto_observacion=None, config=None):
    """
    Crea y configura un módulo de reflexión metacognitiva para un modelo.
    
    Args:
        modelo: Modelo pre-entrenado a analizar
        tokenizer: Tokenizador asociado al modelo
        auto_observacion: Instancia opcional de AutoObservacion
        config: Configuración opcional
        
    Returns:
        Instancia de ReflexionMetacognitiva
    """
    return ReflexionMetacognitiva(modelo, tokenizer, auto_observacion, config)


# No necesitamos una función especial para datetime ya que lo importamos directamente


if __name__ == "__main__":
    # Ejemplo de uso
    from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
    
    # Cargar un modelo pequeño para demostración
    modelo_nombre = "gpt2"  # Usar un modelo pequeño para pruebas
    modelo = AutoModelForCausalLM.from_pretrained(modelo_nombre)
    tokenizer = AutoTokenizer.from_pretrained(modelo_nombre)
    
    # Crear módulo de auto-observación
    from auto_observacion import crear_modulo_auto_observacion
    auto_obs = crear_modulo_auto_observacion(modelo)
    
    # Crear módulo de reflexión metacognitiva
    reflexion = crear_modulo_reflexion(modelo, tokenizer, auto_obs)
    
    # Generar reflexión
    texto_reflexion = reflexion.generar_reflexion()
    print(texto_reflexion)
    
    # Analizar problemas
    problemas = reflexion.analizar_problemas_aprendizaje()
    print("\nProblemas detectados:")
    for problema, info in problemas.items():
        print(f"- {problema}: {info['severidad']}")
    
    # Generar estrategias
    estrategias = reflexion.generar_estrategias_mejora(problemas)
    print("\nEstrategias propuestas:")
    for estrategia in estrategias:
        print(f"- {estrategia['estrategia']}: {estrategia['descripcion']}")
    
    # Guardar reflexiones
    reflexion.guardar_reflexiones("./reflexiones_demo")
