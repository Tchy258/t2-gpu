#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse

def mover_csv(origen, destino):
    """
    Recorre recursivamente la carpeta 'origen', 
    y mueve todos los archivos .csv a la carpeta 'destino'.
    """
    # Asegurarse de que la carpeta de destino exista
    os.makedirs(destino, exist_ok=True)

    # os.walk recorre todas las subcarpetas de forma recursiva
    for carpeta_actual, subcarpetas, archivos in os.walk(origen):
        for nombre_archivo in archivos:
            # Verificamos la extensión (en minúsculas)
            if nombre_archivo.lower().endswith('.csv'):
                ruta_origen = os.path.join(carpeta_actual, nombre_archivo)
                ruta_destino = os.path.join(destino, nombre_archivo)
                
                try:
                    # Mover el archivo al destino
                    shutil.move(ruta_origen, ruta_destino)
                    print(f"Movido: {ruta_origen} -> {ruta_destino}")
                except Exception as e:
                    print(f"Error al mover {ruta_origen}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mueve todos los archivos .csv de una carpeta (y sus subcarpetas) a otra carpeta."
    )
    parser.add_argument(
        "carpeta_origen",
        help="Ruta de la carpeta donde se buscarán los archivos .csv"
    )
    parser.add_argument(
        "carpeta_destino",
        help="Ruta de la carpeta donde se moverán los archivos .csv"
    )
    args = parser.parse_args()

    origen = args.carpeta_origen
    destino = args.carpeta_destino

    mover_csv(origen, destino)
