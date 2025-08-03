#!/bin/bash

echo "🎬 AI-Powered Vertical Video Generator"
echo "======================================"

# Verificar si el entorno virtual existe
if [ ! -d "venv" ]; then
    echo "❌ Entorno virtual no encontrado. Ejecutando configuración inicial..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Configuración completada!"
else
    echo "✅ Entorno virtual encontrado"
fi

# Activar entorno virtual
source venv/bin/activate

echo ""
echo "Selecciona una opción:"
echo "1) Generar videos desde imágenes en carpeta X (enhanced_video_generator.py)"
echo "2) Generar video de demostración (simple_video_generator.py)"
echo "3) Crear imagen de muestra (create_sample_image.py)"
echo "4) Salir"
echo ""

read -p "Ingresa tu opción (1-4): " choice

case $choice in
    1)
        echo "🎬 Procesando imágenes desde carpeta X..."
        python enhanced_video_generator.py
        ;;
    2)
        echo "🎬 Generando video de demostración..."
        python simple_video_generator.py
        ;;
    3)
        echo "🖼️ Creando imagen de muestra..."
        python create_sample_image.py
        ;;
    4)
        echo "👋 ¡Hasta luego!"
        exit 0
        ;;
    *)
        echo "❌ Opción inválida. Usa 1, 2, 3 o 4."
        exit 1
        ;;
esac

echo ""
echo "🎉 ¡Proceso completado!"
echo "📁 Revisa la carpeta Y/ para ver los videos generados"