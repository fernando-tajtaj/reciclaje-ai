import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
import time
import requests

# ------------------------
# CONFIGURACIÓN CUSTOM VISION
# ------------------------
PREDICTION_KEY = "6xxEK0KVqeYvBxtJB2SL7WqHnSqv306q6PX6L7IlYIYMQB45AaKGJQQJ99BEACLArgHXJ3w3AAAIACOGGM6A"
ENDPOINT = "https://resource-umg-recicla-ia.cognitiveservices.azure.com"
PROJECT_ID = "bfebea5f-381e-49df-970a-76ecfbbcea67"
PUBLISHED_NAME = "iteration-umg-recicla-ia-v1"

# ------------------------
# RUTAS
# ------------------------
RUTA_FONDO = "resources/fondo.png"

# ------------------------
# VENTANA PRINCIPAL
# ------------------------
root = tk.Tk()
root.title("RAIZ")
root.geometry("1280x720")
root.resizable(False, False)

# Canvas y fondo
canvas = tk.Canvas(root, width=1280, height=720, highlightthickness=0)
canvas.pack()
bg_img = Image.open(RUTA_FONDO).resize((1280, 720))
bg_photo = ImageTk.PhotoImage(bg_img)
canvas.create_image(0, 0, anchor="nw", image=bg_photo)

# Label para cámara
cam_label = tk.Label(root, bd=0, highlightthickness=0, relief="flat", bg="#000000")
canvas.create_window(150, 140, anchor="nw", window=cam_label, width=690, height=410)

# Label para miniatura de la captura
captura_label = tk.Label(root, bd=0, highlightthickness=0, relief="flat", bg="black")
canvas.create_window(895, 240, anchor="nw", window=captura_label, width=240, height=240)

# Label para predicción
prediccion_label = tk.Label(
    root,
    text="Esperando predicción...",
    font=("Arial", 12, "bold"),
    fg="black",
    bg="#9dd241",
    bd=0,
    highlightthickness=0
)
canvas.create_window(170, 585, anchor="nw", window=prediccion_label, width=940, height=40)

# ------------------------
# VARIABLES DE CÁMARA
# ------------------------
cap = cv2.VideoCapture(0)
captura_actual = None

# ------------------------
# FUNCIONES
# ------------------------

def actualizar_camara():
    global captura_actual
    while True:
        ret, frame = cap.read()
        if ret:
            captura_actual = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb).resize((690, 410))
            img_tk = ImageTk.PhotoImage(img_pil)
            cam_label.config(image=img_tk)
            cam_label.image = img_tk
        time.sleep(0.03)

def esperar_y_capturar():
    time.sleep(5)  # Espera 5 segundos
    if captura_actual is not None:
        # Mostrar miniatura en la GUI
        frame_rgb = cv2.cvtColor(captura_actual, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb).resize((278, 280))
        img_tk = ImageTk.PhotoImage(img_pil)
        captura_label.config(image=img_tk)
        captura_label.image = img_tk

        # Enviar imagen directamente desde memoria
        _, buffer = cv2.imencode('.jpg', captura_actual)
        enviar_a_custom_vision(buffer.tobytes())

def enviar_a_custom_vision(imagen_bytes):
    url = f"{ENDPOINT}/customvision/v3.0/Prediction/{PROJECT_ID}/classify/iterations/{PUBLISHED_NAME}/image"

    headers = {
        "Prediction-Key": PREDICTION_KEY,
        "Content-Type": "application/octet-stream"
    }

    try:
        response = requests.post(url, headers=headers, data=imagen_bytes)

        if response.status_code == 200:
            try:
                resultado = response.json()
                mostrar_prediccion(resultado)
            except ValueError as e:
                print("Respuesta no es JSON válida:", response.text)
                prediccion_label.config(text="Respuesta inválida del servidor")
        else:
            print("Error:", response.status_code)
            print("Contenido de error:", response.text)
            prediccion_label.config(text=f"Error al predecir ({response.status_code})")

    except Exception as e:
        print("Excepción:", e)
        prediccion_label.config(text="Error al enviar imagen")


def mostrar_prediccion(resultado):
    if "predictions" in resultado and len(resultado["predictions"]) > 0:
        pred = resultado["predictions"][0]
        etiqueta = pred["tagName"]
        probabilidad = pred["probability"]
        texto = f"Predicción: {etiqueta} ({probabilidad * 100:.2f}%)"
        prediccion_label.config(text=texto)
    else:
        prediccion_label.config(text="No se detectó ninguna predicción")

# ------------------------
# INICIAR HILOS
# ------------------------
threading.Thread(target=actualizar_camara, daemon=True).start()
threading.Thread(target=esperar_y_capturar, daemon=True).start()

# ------------------------
# MAINLOOP
# ------------------------
root.mainloop()
cap.release()