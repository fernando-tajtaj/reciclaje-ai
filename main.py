import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
import time
import requests
import serial
from tabulate import tabulate

PUERTO_SERIAL = "COM5"
BAUD_RATE = 9600

PREDICTION_KEY = "6xxEK0KVqeYvBxtJB2SL7WqHnSqv306q6PX6L7IlYIYMQB45AaKGJQQJ99BEACLArgHXJ3w3AAAIACOGGM6A"
ENDPOINT = "https://resource-umg-recicla-ia.cognitiveservices.azure.com"
PROJECT_ID = "bfebea5f-381e-49df-970a-76ecfbbcea67"
PUBLISHED_NAME = "iteration-umg-recicla-ia-v1"

RUTA_FONDO = "resources/fondo.png"

registro_predicciones = []
prediccion_contador = 0
ejecutar = True

try:
    arduino = serial.Serial(PUERTO_SERIAL, BAUD_RATE, timeout=1)
    time.sleep(2)
except Exception as e:
    print(f"No se pudo conectar con Arduino en {PUERTO_SERIAL}: {e}")
    arduino = None

root = tk.Tk()
root.title("RAIZ")
root.geometry("1280x720")
root.resizable(False, False)

canvas = tk.Canvas(root, width=1280, height=720, highlightthickness=0)
canvas.pack()
bg_img = Image.open(RUTA_FONDO).resize((1280, 720))
bg_photo = ImageTk.PhotoImage(bg_img)
canvas.create_image(0, 0, anchor="nw", image=bg_photo)

cam_label = tk.Label(root, bd=0, highlightthickness=0, relief="flat", bg="#000000")
canvas.create_window(150, 140, anchor="nw", window=cam_label, width=690, height=410)

captura_label = tk.Label(root, bd=0, highlightthickness=0, relief="flat", bg="black")
canvas.create_window(895, 240, anchor="nw", window=captura_label, width=240, height=240)

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

cap = cv2.VideoCapture(0)
captura_actual = None

pred_temp = {}

def enviar_senal_arduino(categoria):
    if arduino is None:
        print("Arduino no conectado")
        return

    codigo = {
        "plastic": "P",
        "cardboard": "C",
        "paper": "R",
        "none": "N"
    }

    señal = codigo.get(categoria, "N")
    try:
        arduino.write(señal.encode())
        pred_temp["mensaje_final"] = f"Enviado a Arduino: {señal}"
    except Exception as e:
        print(f"Error enviando a Arduino: {e}")

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

def capturar_y_enviar():
    global captura_actual
    if captura_actual is not None:
        frame_rgb = cv2.cvtColor(captura_actual, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb).resize((278, 280))
        img_tk = ImageTk.PhotoImage(img_pil)
        captura_label.config(image=img_tk)
        captura_label.image = img_tk

        _, buffer = cv2.imencode('.jpg', captura_actual)
        enviar_a_custom_vision(buffer.tobytes())
    else:
        print("Paso algo")

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
            except ValueError:
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
    global prediccion_contador
    if "predictions" in resultado and len(resultado["predictions"]) > 0:
        pred = resultado["predictions"][0]
        etiqueta = pred["tagName"].lower()
        probabilidad = pred["probability"]
        texto = f"Predicción: {etiqueta} ({probabilidad * 100:.2f}%)"
        prediccion_label.config(text=texto)

        pred_temp["etiqueta"] = etiqueta
        pred_temp["probabilidad"] = f"{probabilidad * 100:.2f}%"

        if etiqueta in ("plastic", "cardboard", "paper") and probabilidad > 0.5:
            enviar_senal_arduino(etiqueta)
        else:
            enviar_senal_arduino("none")
    else:
        prediccion_label.config(text="No se detectó ninguna predicción")
        enviar_senal_arduino("none")
        pred_temp["etiqueta"] = "ninguna"
        pred_temp["probabilidad"] = "0.00%"

    # Guardar predicción
    prediccion_contador += 1
    registro_predicciones.append({
        "Iteración": prediccion_contador,
        "Etiqueta": pred_temp.get("etiqueta", ""),
        "Probabilidad": pred_temp.get("probabilidad", ""),
        "Mensaje Inicial": pred_temp.get("mensaje_inicial", "N/A"),
        "Mensaje": pred_temp.get("mensaje", "N/A"),
        "Mensaje Final": pred_temp.get("mensaje_final", "N/A")
    })

    print("\nHistorial de predicciones:")
    print(tabulate(registro_predicciones, headers="keys", tablefmt="grid"))

def escuchar_arduino():
    while True:
        if not ejecutar:
            time.sleep(0.1)
            continue
        if arduino and arduino.in_waiting > 0:
            linea = arduino.readline().decode(errors='ignore').strip()
            if linea == "READY":
                pred_temp.clear()
                pred_temp["mensaje_inicial"] = "READY"
                time.sleep(3)
                capturar_y_enviar()

            elif linea == "PROCESING":
                pred_temp["mensaje"] = "PROCESING"
        else:
            time.sleep(0.5)

threading.Thread(target=actualizar_camara, daemon=True).start()
threading.Thread(target=escuchar_arduino, daemon=True).start()

root.mainloop()
cap.release()