import pytest
from fastapi.testclient import TestClient

# Importamos la app de FastAPI desde el backend
from src.api.main_api import app

# Creamos un cliente de simulación. Esto levantará la aplicación (cargará los modelos en memoria)
# igual que si hiciéramos un 'fastapi dev', pero de forma temporal para los tests.
client = TestClient(app)

def test_status_endpoint():
    """Verifica que el estado básico del servidor responde correctamente."""
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "Backend funcionando correctamente"}


def test_recomendar_svd_schema():
    """
    Prueba que el endpoint del modelo SVD (nuestro modelo estrella) devuelva 
    el esquema correcto de datos (RecommendationResponse) usando Pydantic por debajo.
    """
    # Pedimos 5 recomendaciones para el usuario 1
    response = client.get("/recomendar/svd/1?n=5")
    
    # Si la API devuelve un 503, significa que el modelo no estaba en la carpeta.
    # Así el test avisa de que se saltó, en vez de marcar un fallo rojo falso.
    if response.status_code == 503:
        pytest.skip("Modelo SVD o CSV de ratings no encontrados en disco. Test omitido.")

    # 1. Verificamos que responde OK
    assert response.status_code == 200
    
    datos = response.json()
    
    # 2. Verificamos las claves del contrato de la API (RecommendationResponse)
    assert "recomendaciones" in datos, "Falta la lista de recomendaciones en la respuesta"
    assert "modelo" in datos, "Falta el nombre del modelo en la respuesta"
    
    # 3. Verificamos la longitud
    assert len(datos["recomendaciones"]) <= 5
    
    # 4. Verificamos las propiedades de una recomendación aleatoria
    if len(datos["recomendaciones"]) > 0:
        primera_peli = datos["recomendaciones"][0]
        assert "tmdb_id" in primera_peli
        assert "predicted_rating" in primera_peli
        assert "titulo" in primera_peli
        assert "poster_path" in primera_peli


def test_recomendar_wnd_schema():
    """
    Prueba el endpoint de Wide & Deep ONNX.
    """
    response = client.get("/recomendar/wnd/1?n=2")
    
    if response.status_code == 503:
        pytest.skip("Archivos de Wide&Deep (ONNX o Mapeos) no disponibles. Test omitido.")

    assert response.status_code == 200
    datos = response.json()
    assert "recomendaciones" in datos
    
    
def test_recomendar_content_cold_start():
    """
    Prueba que si a TF-IDF le pasamos un usuario inexistente (ej. ID 9999999), 
    devuelva la estrategia 'Cold Start Populares' sin fallar.
    """
    response = client.get("/recomendar/content/9999999?n=3")
    
    if response.status_code == 503:
        pytest.skip("Modelo TF-IDF no disponible en local.")
        
    assert response.status_code == 200
    datos = response.json()
    assert "recomendaciones" in datos
    
    # Debería devolver pelis con rating predeterminado de 4.5
    if len(datos["recomendaciones"]) > 0:
        assert datos["recomendaciones"][0]["predicted_rating"] == 4.5
