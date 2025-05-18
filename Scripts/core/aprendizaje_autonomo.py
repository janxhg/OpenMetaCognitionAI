#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sistema de Aprendizaje Autónomo para modelos de lenguaje.
Este script implementa un enfoque donde el modelo aprende de forma autónoma,
explorando, reflexionando y mejorando por sí mismo, similar a un estudiante humano.
"""

import os
import json
import logging
import argparse
import torch
import numpy as np
import random
import time
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class MemoriaEstudio:
    """Clase para gestionar la memoria de estudio del modelo."""
    
    def __init__(self, ruta_memoria=None):
        """Inicializa la memoria de estudio."""
        self.conceptos = {}  # Diccionario de conceptos aprendidos
        self.conexiones = {}  # Conexiones entre conceptos
        self.sesiones_estudio = []  # Historial de sesiones de estudio
        self.reflexiones = []  # Reflexiones sobre el aprendizaje
        
        # Cargar memoria existente si se proporciona ruta
        if ruta_memoria and os.path.exists(ruta_memoria):
            self.cargar_memoria(ruta_memoria)
    
    def agregar_concepto(self, nombre, definicion, dominio, dificultad=0, importancia=0):
        """Agrega un nuevo concepto a la memoria."""
        if nombre in self.conceptos:
            # Actualizar concepto existente
            self.conceptos[nombre]["definicion"] = definicion
            self.conceptos[nombre]["dominio"] = dominio
            self.conceptos[nombre]["revisiones"] += 1
            self.conceptos[nombre]["ultima_revision"] = datetime.now().isoformat()
        else:
            # Crear nuevo concepto
            self.conceptos[nombre] = {
                "definicion": definicion,
                "dominio": dominio,
                "dificultad": dificultad,  # 0-10, qué tan difícil es de entender
                "importancia": importancia,  # 0-10, qué tan importante es
                "familiaridad": 0,  # 0-10, qué tan familiar es
                "revisiones": 1,
                "fecha_aprendizaje": datetime.now().isoformat(),
                "ultima_revision": datetime.now().isoformat()
            }
    
    def conectar_conceptos(self, concepto1, concepto2, tipo_relacion, fuerza=5):
        """Establece una conexión entre dos conceptos."""
        if concepto1 not in self.conceptos or concepto2 not in self.conceptos:
            return False
        
        clave_conexion = f"{concepto1}|{concepto2}"
        
        if clave_conexion in self.conexiones:
            # Actualizar conexión existente
            self.conexiones[clave_conexion]["tipo"] = tipo_relacion
            self.conexiones[clave_conexion]["fuerza"] += 1
            self.conexiones[clave_conexion]["ultima_activacion"] = datetime.now().isoformat()
        else:
            # Crear nueva conexión
            self.conexiones[clave_conexion] = {
                "conceptos": [concepto1, concepto2],
                "tipo": tipo_relacion,
                "fuerza": fuerza,  # 0-10, qué tan fuerte es la conexión
                "fecha_creacion": datetime.now().isoformat(),
                "ultima_activacion": datetime.now().isoformat()
            }
        
        return True
    
    def registrar_sesion_estudio(self, material, duracion, conceptos_estudiados, calidad_aprendizaje):
        """Registra una sesión de estudio."""
        sesion = {
            "fecha": datetime.now().isoformat(),
            "material": material,
            "duracion_minutos": duracion,
            "conceptos_estudiados": conceptos_estudiados,
            "calidad_aprendizaje": calidad_aprendizaje,  # 0-10
            "estado_mental": {
                "concentracion": random.randint(0, 10),
                "motivacion": random.randint(0, 10),
                "cansancio": random.randint(0, 10)
            }
        }
        
        self.sesiones_estudio.append(sesion)
    
    def agregar_reflexion(self, texto, tipo="general"):
        """Agrega una reflexión sobre el proceso de aprendizaje."""
        reflexion = {
            "fecha": datetime.now().isoformat(),
            "tipo": tipo,  # "general", "concepto", "conexion", "estrategia"
            "texto": texto
        }
        
        self.reflexiones.append(reflexion)
    
    def obtener_conceptos_por_dominio(self, dominio):
        """Obtiene todos los conceptos de un dominio específico."""
        return {k: v for k, v in self.conceptos.items() if v["dominio"] == dominio}
    
    def obtener_conceptos_para_repaso(self, n=5):
        """Obtiene conceptos que necesitan repaso según la curva del olvido."""
        # Implementación simple de espaciado de repaso
        ahora = datetime.now()
        conceptos_con_tiempo = []
        
        for nombre, datos in self.conceptos.items():
            ultima_revision = datetime.fromisoformat(datos["ultima_revision"])
            tiempo_pasado = (ahora - ultima_revision).total_seconds() / 86400  # días
            
            # Factor de olvido basado en revisiones previas y dificultad
            factor_olvido = (datos["dificultad"] + 1) / (datos["revisiones"] + 1)
            
            # Prioridad de repaso
            prioridad = tiempo_pasado * factor_olvido * (datos["importancia"] + 1)
            
            conceptos_con_tiempo.append((nombre, prioridad))
        
        # Ordenar por prioridad de repaso
        conceptos_con_tiempo.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver los N conceptos con mayor prioridad
        return [self.conceptos[nombre] for nombre, _ in conceptos_con_tiempo[:n]]
    
    def generar_mapa_conceptual(self):
        """Genera un mapa conceptual basado en los conceptos y conexiones."""
        mapa = {
            "nodos": [],
            "enlaces": []
        }
        
        # Agregar nodos (conceptos)
        for nombre, datos in self.conceptos.items():
            mapa["nodos"].append({
                "id": nombre,
                "label": nombre,
                "dominio": datos["dominio"],
                "importancia": datos["importancia"]
            })
        
        # Agregar enlaces (conexiones)
        for clave, datos in self.conexiones.items():
            c1, c2 = datos["conceptos"]
            mapa["enlaces"].append({
                "source": c1,
                "target": c2,
                "label": datos["tipo"],
                "value": datos["fuerza"]
            })
        
        return mapa
    
    def guardar_memoria(self, ruta_archivo):
        """Guarda la memoria de estudio en un archivo JSON."""
        memoria = {
            "conceptos": self.conceptos,
            "conexiones": self.conexiones,
            "sesiones_estudio": self.sesiones_estudio,
            "reflexiones": self.reflexiones,
            "fecha_guardado": datetime.now().isoformat()
        }
        
        with open(ruta_archivo, 'w', encoding='utf-8') as f:
            json.dump(memoria, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Memoria guardada en {ruta_archivo}")
    
    def cargar_memoria(self, ruta_archivo):
        """Carga la memoria de estudio desde un archivo JSON."""
        try:
            with open(ruta_archivo, 'r', encoding='utf-8') as f:
                memoria = json.load(f)
            
            self.conceptos = memoria.get("conceptos", {})
            self.conexiones = memoria.get("conexiones", {})
            self.sesiones_estudio = memoria.get("sesiones_estudio", [])
            self.reflexiones = memoria.get("reflexiones", [])
            
            logger.info(f"Memoria cargada desde {ruta_archivo}")
            logger.info(f"Conceptos: {len(self.conceptos)}, Conexiones: {len(self.conexiones)}")
            
            return True
        except Exception as e:
            logger.error(f"Error al cargar memoria: {str(e)}")
            return False

class BibliotecaConocimiento:
    """Clase para gestionar la biblioteca de conocimiento disponible para estudio."""
    
    def __init__(self, ruta_biblioteca=None):
        """Inicializa la biblioteca de conocimiento."""
        self.materiales = {}  # Materiales de estudio por dominio
        self.dominios = set()  # Dominios de conocimiento
        
        # Cargar biblioteca existente si se proporciona ruta
        if ruta_biblioteca and os.path.exists(ruta_biblioteca):
            self.cargar_biblioteca(ruta_biblioteca)
    
    def agregar_material(self, titulo, contenido, dominio, dificultad=5, prerequisitos=None):
        """Agrega un nuevo material de estudio a la biblioteca."""
        if titulo in self.materiales:
            logger.warning(f"El material '{titulo}' ya existe en la biblioteca")
            return False
        
        self.materiales[titulo] = {
            "contenido": contenido,
            "dominio": dominio,
            "dificultad": dificultad,  # 0-10
            "prerequisitos": prerequisitos or [],
            "fecha_agregado": datetime.now().isoformat()
        }
        
        self.dominios.add(dominio)
        return True
    
    def obtener_material(self, titulo):
        """Obtiene un material de estudio por su título."""
        return self.materiales.get(titulo)
    
    def obtener_materiales_por_dominio(self, dominio):
        """Obtiene todos los materiales de un dominio específico."""
        return {k: v for k, v in self.materiales.items() if v["dominio"] == dominio}
    
    def obtener_materiales_por_dificultad(self, min_dificultad, max_dificultad):
        """Obtiene materiales dentro de un rango de dificultad."""
        return {k: v for k, v in self.materiales.items() 
                if min_dificultad <= v["dificultad"] <= max_dificultad}
    
    def recomendar_material(self, memoria_estudio, dominio=None, max_dificultad=None):
        """Recomienda material de estudio basado en el estado actual de la memoria."""
        # Filtrar por dominio si se especifica
        candidatos = self.materiales
        if dominio:
            candidatos = self.obtener_materiales_por_dominio(dominio)
        
        # Filtrar por dificultad si se especifica
        if max_dificultad is not None:
            candidatos = {k: v for k, v in candidatos.items() if v["dificultad"] <= max_dificultad}
        
        # Si no hay candidatos, devolver None
        if not candidatos:
            return None
        
        # Evaluar cada material según criterios de aprendizaje
        materiales_puntuados = []
        
        for titulo, datos in candidatos.items():
            # Verificar si se cumplen los prerequisitos
            prerequisitos_cumplidos = True
            for prereq in datos["prerequisitos"]:
                if prereq not in memoria_estudio.conceptos:
                    prerequisitos_cumplidos = False
                    break
            
            if not prerequisitos_cumplidos:
                continue
            
            # Calcular novedad (preferir material que introduce conceptos nuevos)
            conceptos_nuevos = 0
            # Aquí se podría implementar un análisis del contenido para identificar conceptos
            
            # Calcular relevancia para el estado actual de conocimiento
            relevancia = random.randint(1, 10)  # Simplificado para este ejemplo
            
            # Puntuación final
            puntuacion = relevancia + conceptos_nuevos
            
            materiales_puntuados.append((titulo, puntuacion))
        
        # Ordenar por puntuación
        materiales_puntuados.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver el mejor material o None si no hay candidatos válidos
        if materiales_puntuados:
            mejor_titulo = materiales_puntuados[0][0]
            return self.materiales[mejor_titulo]
        
        return None
    
    def guardar_biblioteca(self, ruta_archivo):
        """Guarda la biblioteca de conocimiento en un archivo JSON."""
        biblioteca = {
            "materiales": self.materiales,
            "dominios": list(self.dominios),
            "fecha_guardado": datetime.now().isoformat()
        }
        
        with open(ruta_archivo, 'w', encoding='utf-8') as f:
            json.dump(biblioteca, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Biblioteca guardada en {ruta_archivo}")
    
    def cargar_biblioteca(self, ruta_archivo):
        """Carga la biblioteca de conocimiento desde un archivo JSON."""
        try:
            with open(ruta_archivo, 'r', encoding='utf-8') as f:
                biblioteca = json.load(f)
            
            self.materiales = biblioteca.get("materiales", {})
            self.dominios = set(biblioteca.get("dominios", []))
            
            logger.info(f"Biblioteca cargada desde {ruta_archivo}")
            logger.info(f"Materiales: {len(self.materiales)}, Dominios: {len(self.dominios)}")
            
            return True
        except Exception as e:
            logger.error(f"Error al cargar biblioteca: {str(e)}")
            return False
    
    def cargar_desde_directorio(self, ruta_directorio, extension=".txt"):
        """Carga materiales de estudio desde archivos en un directorio."""
        try:
            ruta = Path(ruta_directorio)
            archivos = list(ruta.glob(f"*{extension}"))
            
            for archivo in archivos:
                try:
                    with open(archivo, 'r', encoding='utf-8') as f:
                        contenido = f.read()
                    
                    # Inferir dominio del nombre del directorio padre
                    dominio = archivo.parent.name
                    
                    # Usar nombre de archivo como título
                    titulo = archivo.stem
                    
                    # Inferir dificultad (ejemplo simple)
                    palabras = len(contenido.split())
                    dificultad = min(10, max(1, palabras // 500))
                    
                    self.agregar_material(titulo, contenido, dominio, dificultad)
                    
                except Exception as e:
                    logger.error(f"Error al cargar archivo {archivo}: {str(e)}")
            
            logger.info(f"Cargados {len(archivos)} materiales desde {ruta_directorio}")
            return True
        
        except Exception as e:
            logger.error(f"Error al cargar desde directorio: {str(e)}")
            return False
