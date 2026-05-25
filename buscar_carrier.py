import os
import s3fs
import pandas as pd
from dotenv import load_dotenv

# Cargar variables de entorno (por si las credenciales de AWS están en su archivo .env)
load_dotenv("/home/ec2-user/Inicio_falabella/.env")

# ==================== CONFIGURACIÓN ====================
# Ajuste esta ruta a la ubicación real de sus archivos de Tracking en S3
S3_PATH_TRACKING = "s3://falabella-data/processed/tracking_data/" 

# Nombre o ID del Carrier que desea buscar (puede ser string o número según su base de datos)
CARRIER_A_BUSCAR = "Nombre_O_Codigo_Del_Carrier" 
# =======================================================

def buscar_por_carrier():
    print(f"Iniciando búsqueda de carrier: '{CARRIER_A_BUSCAR}' en {S3_PATH_TRACKING}...", flush=True)
    
    # Conectar a S3 usando s3fs (reutiliza credenciales del sistema o del archivo .env)
    fs = s3fs.S3FileSystem()
    
    # Limpiar la ruta para el buscador de archivos
    bucket_prefix = S3_PATH_TRACKING.replace("s3://", "")
    
    try:
        # Listar todos los archivos .parquet dentro de la ruta (incluyendo subcarpetas si las hay)
        print("Buscando archivos en S3...", flush=True)
        archivos = sorted([f"s3://{f}" for f in fs.glob(f"{bucket_prefix}/**/*.parquet", recursive=True)])
        if not archivos:
            # Intentar búsqueda simple sin subcarpetas si falló la recursiva
            archivos = sorted([f"s3://{f}" for f in fs.glob(f"{bucket_prefix}*.parquet")])
    except Exception as e:
        print(f"Error al conectar o listar S3: {e}")
        return

    if not archivos:
        print("No se encontraron archivos Parquet en la ruta especificada.")
        return

    print(f"Se encontraron {len(archivos)} archivos. Buscando coincidencia de registros...", flush=True)
    
    total_coincidencias = 0
    resultados = []

    for i, archivo in enumerate(archivos, 1):
        try:
            # LEER CON FILTRO DE PYARROW (Solo descarga las filas que coincidan con el carrier)
            # Esto evita que la EC2 se quede sin memoria (OOM)
            df = pd.read_parquet(
                archivo,
                filters=[('carrier', '==', CARRIER_A_BUSCAR)]
            )
            
            if not df.empty:
                print(f"  [{i}/{len(archivos)}] ¡Encontradas {len(df):,} filas en: {archivo.split('/')[-1]}", flush=True)
                resultados.append(df)
                total_coincidencias += len(df)
            
            del df
            
        except Exception as e:
            # En caso de que el archivo no tenga la columna 'carrier', lo omitirá sin caerse
            continue

    print("\n" + "="*50)
    if resultados:
        # Consolidar todos los registros encontrados
        df_final = pd.concat(resultados, ignore_index=True)
        print(f"BÚSQUEDA COMPLETADA. Total filas encontradas: {total_coincidencias:,}")
        print("="*50)
        
        # Mostrar las primeras 10 filas encontradas como muestra
        print("\nMuestra de los datos encontrados:")
        print(df_final.head(10).to_string())
        
        # Opcional: Guardar el resultado en un archivo local en la EC2
        output_file = "resultados_busqueda_carrier.csv"
        df_final.to_csv(output_file, index=False)
        print(f"\nResultados completos guardados localmente en: {output_file}")
    else:
        print(f"No se encontraron registros para el carrier '{CARRIER_A_BUSCAR}'.")
    print("="*50)

if __name__ == "__main__":
    buscar_por_carrier()