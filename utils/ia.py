import re
import requests
from sympy import Eq, symbols, sympify, solve, pi, E

# Reutilizamos sesión HTTP para menos latencia
_session = requests.Session()
BASE_URL = "http://localhost:11434/api/generate"

def _preprocesar(expr: str) -> str:
    """Inserta multiplicaciones implícitas y limpia espacios básicos."""
    e = expr.strip()
    # 2x -> 2*x ; x( -> x*( ; )( -> )*( ; )2 -> )*2
    e = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', e)
    e = re.sub(r'([a-zA-Z])(\()', r'\1*\2', e)
    e = re.sub(r'(\))(\()', r'\1*\2', e)
    e = re.sub(r'(\))(\d)',  r'\1*\2', e)
    # Operadores visibles
    e = e.replace('×', '*').replace('÷', '/').replace('−', '-')
    # Asegurar espacios alrededor de '=' (ayuda a algunos modelos)
    e = re.sub(r'\s*=\s*', ' = ', e)
    return e

def _es_formula(expr: str) -> bool:
    """Detecta si es una fórmula (sin '=') o una ecuación (con '=')."""
    return '=' not in expr

def _tiene_variables(expr: str) -> bool:
    """Detecta si la expresión tiene variables sin valores numéricos."""
    # Variables comunes en fórmulas
    variables = ['x', 'y', 'z', 'r', 'a', 'b', 'h', 'm', 'v', 'g', 't', 'v0']
    expr_lower = expr.lower()
    
    # Verificar si tiene variables Y no son solo coeficientes numéricos
    for var in variables:
        # Buscar la variable como símbolo independiente
        if re.search(rf'\b{var}\b', expr_lower):
            return True
    return False

def resolver_matematicas(expr: str, solo_resultado: bool = False, stream=False):
    """
    - solo_resultado=False: pide PASOS al modelo qwen2.5:1.5b (en español).
    - solo_resultado=True : intenta resolver con Sympy primero (exacto y corto).
                            si falla, pide SOLO el resultado al modelo.
    """
    expr_pp = _preprocesar(expr)

    # Detectar si es una fórmula sin valores
    es_formula = _es_formula(expr_pp)
    tiene_vars = _tiene_variables(expr_pp)

    # Detectar complejidad de la ecuación para ajustar tokens dinámicamente
    complejidad = len(expr_pp) + expr_pp.count('^') * 10 + expr_pp.count('sqrt') * 5

    if solo_resultado:
        # CASO 1: Es una FÓRMULA (sin '=') con variables → Explicar qué calcula
        if es_formula and tiene_vars:
            prompt = f"Fórmula {expr_pp}: explica en 1 línea"
            r = _session.post(BASE_URL, json={
                "model": "qwen2.5:1.5b",
                "prompt": prompt,
                "options": {
                    "num_predict": 25,
                    "temperature": 0,
                    "top_p": 0.9,
                    "num_ctx": 256,
                    "top_k": 10
                },
                "stream": False,
                "keep_alive": "10m"
            }, timeout=120)
            return (r.json().get("response", "") or "").strip() or "📐 Fórmula sin valores numéricos"
        
        # CASO 2: Es una ECUACIÓN (con '=') → Intentar resolver con Sympy
        try:
            if '=' in expr_pp:
                left, right = expr_pp.split('=', 1)
                eq = Eq(sympify(left), sympify(right))
                # Deducción simple de variables (x,y,z)
                vars_guess = sorted({str(s) for s in eq.free_symbols})
                syms = symbols(vars_guess) if vars_guess else symbols('x')
                sol = solve(eq, *([syms] if isinstance(syms, symbols) else syms), dict=True)
                if sol:
                    # Devuelve el primer valor encontrado (formato corto)
                    k = list(sol[0].keys())[0]
                    valor = sol[0][k]
                    # Formatear valor (si es pi, e, etc.)
                    if hasattr(valor, 'evalf'):
                        valor_num = valor.evalf()
                        return f"{k} = {valor_num}"
                    return f"{k} = {valor}"
            else:
                # Expresión numérica: evalúa
                val = sympify(expr_pp).evalf()
                return str(val)
        except Exception:
            pass  # Si Sympy no puede, caemos al LLM

        # CASO 3: Sympy falló → Pedir al LLM solo el resultado
        prompt = f"Resuelve {expr_pp} = "
        r = _session.post(BASE_URL, json={
            "model": "qwen2.5:1.5b",
            "prompt": prompt,
            "options": {
                "num_predict": 15,
                "temperature": 0,
                "top_p": 0.9,
                "num_ctx": 128,
                "top_k": 5
            },
            "stream": False,
            "keep_alive": "10m"
        }, timeout=120)
        txt = (r.json().get("response", "") or "").strip()
        # Extraer primer número del texto por si el modelo agrega palabras
        nums = re.findall(r'-?\d+(?:[.,]\d+)?', txt)
        return nums[0].replace(',', '.') if nums else (txt or "🛑 No reconozco un número.")
    
    else:
        # MODO: Ver Pasos (explicación detallada)

        # CASO A: Es una FÓRMULA (sin '=') → Explicar con ejemplo
        if es_formula and tiene_vars:
            prompt = f"Fórmula {expr_pp}: explica brevemente con ejemplo.\nRespuesta completa:"
            num_tokens = 100
            num_ctx = 512
        # CASO B: Es una ECUACIÓN (con '=') → Resolver paso a paso (DINÁMICO)
        else:
            # Aceptar explicaciones pedagógicas COMPLETAS (sin cortes)
            # Ajustar tokens según complejidad para explicaciones completas
            if complejidad < 20:
                # Ecuación simple (ej: 2x+3=7)
                num_tokens = 300
                num_ctx = 1536
                prompt = f"Resuelve paso a paso: {expr_pp}"
            elif complejidad < 50:
                # Ecuación mediana (ej: x^2-5x+6=0)
                num_tokens = 400
                num_ctx = 2048
                prompt = f"Resuelve paso a paso: {expr_pp}"
            else:
                # Ecuación compleja/larga (ej: 2x^2+3x-5=sqrt(x+1))
                num_tokens = 500
                num_ctx = 2048
                prompt = f"Resuelve paso a paso: {expr_pp}"

        r = _session.post(BASE_URL, json={
            "model": "qwen2.5:1.5b",
            "prompt": prompt,
            "options": {
                "num_predict": num_tokens,
                "temperature": 0.2,
                "top_p": 0.9,
                "num_ctx": num_ctx,
                "top_k": 20,
                "repeat_penalty": 1.0,
                "stop": ["###", "\n\n\n\n"]
            },
            "stream": stream,
            "keep_alive": "10m"
        }, timeout=150, stream=stream)

        if stream:
            return r
        else:
            resp = (r.json().get("response", "") or "").strip()
            return resp or "🛑 No pude resolver la ecuación."

def explicar_tema_general(pregunta: str, stream=False):
    """
    Explicación de cualquier tema educativo (Física, Química, Biología,
    Historia, Geografía, Literatura, Arte, etc.)

    Por defecto: respuestas de ~25 palabras (cortas pero completas)
    """
    # Detectar si el usuario pide explicación detallada/larga/completa
    import re
    if re.search(r'\b(detallad[oa]|complet[oa]|larg[oa]|explicado|ampli[oa])\b', pregunta.lower()):
        # Usuario pide más detalle
        num_tokens = 100
        prompt = f"{pregunta}\nExplica con detalle:"
    else:
        # POR DEFECTO: respuesta corta (~25 palabras) - MÁS RÁPIDO
        num_tokens = 40
        prompt = f"{pregunta}\nRespuesta breve en 2 frases:"

    r = _session.post(BASE_URL, json={
        "model": "qwen2.5:1.5b",
        "prompt": prompt,
        "options": {
            "num_predict": num_tokens,
            "temperature": 0.2,
            "top_p": 0.9,
            "num_ctx": 512,
            "top_k": 20,
            "stop": ["\n\n", "Pregunta:", "P:"]
        },
        "stream": stream,
        "keep_alive": "10m"
    }, timeout=120, stream=stream)

    if stream:
        return r  # Devuelve el stream directamente
    else:
        return (r.json().get("response", "") or "").strip() or "🛑 No pude explicar el tema."
