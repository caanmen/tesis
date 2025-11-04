from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from utils.ia import resolver_matematicas, explicar_tema_general
import time
import json

app = Flask(__name__)

print("🔥 Precalentando modelo qwen2.5:1.5b...")
import requests
try:
    requests.post("http://localhost:11434/api/generate", json={
        "model": "qwen2.5:1.5b",
        "prompt": "hi",
        "stream": False,
        "keep_alive": "10m",
        "options": {"num_predict": 5, "num_ctx": 128}
    }, timeout=60)
    print("✅ Modelo cargado y listo (se mantendrá en memoria 10 min)")
except Exception as e:
    print(f"⚠️ Advertencia: No se pudo precargar modelo: {e}")

historial = []

@app.route("/")
def index():
    """Renderiza solo el HTML inicial (rápido)"""
    return render_template("index.html", historial=historial[::-1])

@app.route("/api/consulta", methods=["POST"])
def consulta_api():
    """Endpoint API que responde solo con JSON (rápido)"""
    start = time.time()

    try:
        data = request.get_json()
        metodo = data.get("metodo")
        pregunta = data.get("pregunta", "").strip()
        ecuacion = data.get("ecuacion", "").strip()

        if metodo == "explicar":
            if pregunta:
                r = explicar_tema_general(pregunta)
                tipo = "Tema General"
                pregunta_texto = pregunta
            else:
                return jsonify({"error": "Escribe una pregunta."}), 400

        elif metodo == "mate_pasos":
            if ecuacion:
                r = resolver_matematicas(ecuacion, solo_resultado=False)
                tipo = "Matemáticas"
                pregunta_texto = ecuacion
            else:
                return jsonify({"error": "Escribe tu ecuación."}), 400

        elif metodo == "mate_res":
            if ecuacion:
                r = resolver_matematicas(ecuacion, solo_resultado=True)
                tipo = "Matemáticas"
                pregunta_texto = ecuacion
            else:
                return jsonify({"error": "Escribe tu ecuación."}), 400
        else:
            return jsonify({"error": "Método inválido."}), 400

        historial.append({
            "tipo": tipo,
            "pregunta": pregunta_texto,
            "respuesta": r
        })

        elapsed = time.time() - start
        print(f"⏱️ Tiempo total: {elapsed:.2f}s")

        return jsonify({
            "tipo": tipo,
            "pregunta": pregunta_texto,
            "respuesta": r,
            "tiempo": f"{elapsed:.1f}s"
        })

    except Exception as e:
        print(f"❌ Error en consulta: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error del servidor: {str(e)}"}), 500

@app.route("/api/consulta_stream", methods=["POST"])
def consulta_stream():
    """Endpoint con streaming para respuestas en tiempo real"""
    try:
        data = request.get_json()
        metodo = data.get("metodo")
        pregunta = data.get("pregunta", "").strip()
        ecuacion = data.get("ecuacion", "").strip()

        def generate():
            try:
                if metodo == "explicar":
                    if not pregunta:
                        yield f"data: {json.dumps({'error': 'Escribe una pregunta.'})}\n\n"
                        return
                    r = explicar_tema_general(pregunta, stream=True)
                    tipo = "Tema General"
                    pregunta_texto = pregunta

                elif metodo == "mate_pasos":
                    if not ecuacion:
                        yield f"data: {json.dumps({'error': 'Escribe tu ecuación.'})}\n\n"
                        return
                    r = resolver_matematicas(ecuacion, solo_resultado=False, stream=True)
                    tipo = "Matemáticas"
                    pregunta_texto = ecuacion

                elif metodo == "mate_res":
                    if not ecuacion:
                        yield f"data: {json.dumps({'error': 'Escribe tu ecuación.'})}\n\n"
                        return
                    r = resolver_matematicas(ecuacion, solo_resultado=True, stream=True)
                    tipo = "Matemáticas"
                    pregunta_texto = ecuacion
                else:
                    yield f"data: {json.dumps({'error': 'Método inválido.'})}\n\n"
                    return

                yield f"data: {json.dumps({'tipo': tipo, 'pregunta': pregunta_texto, 'start': True})}\n\n"

                if isinstance(r, str):
                    respuesta_completa = r
                    yield f"data: {json.dumps({'chunk': respuesta_completa})}\n\n"
                    historial.append({
                        "tipo": tipo,
                        "pregunta": pregunta_texto,
                        "respuesta": respuesta_completa
                    })
                    yield f"data: {json.dumps({'done': True, 'respuesta': respuesta_completa})}\n\n"
                else:
                    respuesta_completa = ""
                    for line in r.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            if 'response' in chunk:
                                texto = chunk['response']
                                respuesta_completa += texto
                                yield f"data: {json.dumps({'chunk': texto})}\n\n"

                            if chunk.get('done', False):
                                historial.append({
                                    "tipo": tipo,
                                    "pregunta": pregunta_texto,
                                    "respuesta": respuesta_completa
                                })
                                yield f"data: {json.dumps({'done': True, 'respuesta': respuesta_completa})}\n\n"
                                break

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(stream_with_context(generate()), mimetype='text/event-stream')

    except Exception as e:
        print(f"❌ Error en streaming: {str(e)}")
        return jsonify({"error": f"Error del servidor: {str(e)}"}), 500

if __name__ == "__main__":
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(host="0.0.0.0", port=5000, debug=True)
