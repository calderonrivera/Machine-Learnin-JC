import random
import pandas as pd

# Posibles asuntos de correos
asuntos = [
    "Reunión mañana",
    "Oferta especial",
    "Actualización de cuenta",
    "Recordatorio de cita",
    "Factura pendiente",
    "Promoción exclusiva",
    "Información importante",
    "Saludos cordiales",
    "Aviso de seguridad",
    "Novedades del mes"
]

# Generador de dataset de correos con 10 features
def generar_dataset(n=1000):
    data = []
    for i in range(n):
        # Crear remitente ficticio
        remitente = f"usuario{i}@mail.com"
        asunto = random.choice(asuntos)

        # Features aleatorios
        longitud = random.randint(20, 500)  # longitud en caracteres
        num_palabras = random.randint(5, 100)
        tiene_link = random.randint(0, 1)
        tiene_dinero = random.randint(0, 1)
        tiene_premio = random.randint(0, 1)
        tiene_saludo = random.randint(0, 1)
        mayusculas_ratio = round(random.uniform(0, 1), 2)
        signos_exclamacion = random.randint(0, 10)
        num_adjuntos = random.randint(0, 3)
        es_largo = 1 if num_palabras > 50 else 0

        data.append([
            remitente, asunto, longitud, num_palabras, tiene_link,
            tiene_dinero, tiene_premio, tiene_saludo, mayusculas_ratio,
            signos_exclamacion, num_adjuntos, es_largo
        ])

    # Crear DataFrame
    df = pd.DataFrame(data, columns=[
        "remitente", "asunto", "longitud", "num_palabras", "tiene_link",
        "tiene_dinero", "tiene_premio", "tiene_saludo",
        "mayusculas_ratio", "signos_exclamacion",
        "num_adjuntos", "es_largo"
    ])

    return df

# Generar y guardar
df = generar_dataset(1000)
output_file = "cor_features.csv"
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"✅ Dataset generado con {len(df)} correos, 10 features + remitente y asunto. Guardado en {output_file}")
