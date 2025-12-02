import httpx
import asyncio
import chainlit as cl
from src.logger import logger
from src.settings import app_settings

FASTAPI_SERVICE_URL = app_settings.chainlit.fastapi_service_url

async def check_fastapi_health():
    """Проверка доступности FastAPI сервера"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{FASTAPI_SERVICE_URL}/health", timeout=10.0)
            if response.status_code == 200:
                return True
    except Exception as e:
        logger.error(f"Не удалось подключиться к FastAPI: {e}")
    return False


    
@cl.password_auth_callback
def auth_callback(login: str):
    return cl.User(
        identifier=login,
        metadata={
            "role": "user",
            "provider": "open_registration",
            "system": "chainlit_ui"
        }
    )


@cl.on_chat_start
async def on_chat_start():
    """Обработчик начала чата"""
    await cl.Message(content="Привет! Я RAG-ассистент для статей Habr. Подождите пару минут, я загружаю данные.").send()
    
    max_retries = 15
    for i in range(max_retries):
        if await check_fastapi_health():
            break
        await asyncio.sleep(2)
        if i == max_retries - 1:
            await cl.Message(content="Ошибка: FastAPI сервер недоступен. Пожалуйста, убедитесь, что сервер запущен.").send()
            return
        await cl.Message(content=f"Ожидание запуска сервера... ({i+1}/{max_retries})").send()

    cl.user_session.set("session_id", cl.context.session.id)
    await cl.Message(content="Система готова, задайте ваш вопрос! ").send()


@cl.on_message
async def on_message(message: cl.Message):
    """Обработчик сообщений"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{FASTAPI_SERVICE_URL}/answer",
                json={"query": message.content},
                timeout=60.0
            )
            if response.status_code == 200:
                data = response.json()

                answer = data["answer"]
                buttons = [(source['document_id'], source['url']) for source in data["sources"]] 
                logger.info(buttons)
                actions = [
                    cl.Action(
                        name="button",
                        payload={"num_button": num_button+1, "document_id": document_id, 'url': url},
                        label=str(num_button+1)
                    )
                    for num_button, (document_id, url) in enumerate(buttons)
                ]
            else:
                answer = f"Ошибка при получении ответа от сервера (статус: {response.status_code})"
                actions = []
    except httpx.TimeoutException:
        answer = "Таймаут при ожидании ответа от сервера"
        actions = []
    except Exception as e:
        logger.error(f"Ошибка при запросе к FastAPI: {e}")
        answer = f"Ошибка при подключении к серверу: {str(e)}"
        actions = []

    await cl.Message(content=answer, actions=actions).send()


@cl.action_callback("button")
async def handle_button(action):
    document_id = action.payload["document_id"]
    num_button = action.payload["num_button"]
    url = action.payload.get("url", "")
    url_part = f"(URL {url})" if url else ""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{FASTAPI_SERVICE_URL}/summarize",
                json={"document_id": document_id},
                timeout=40.0
            )
            if response.status_code == 200:
                data = response.json()
                summary = data["summary"]
                await cl.Message(
                    content=f"Суммаризация статьи №{num_button} {url_part}:\n{summary}"
                ).send()
            else:
                await cl.Message(content=f"Ошибка при получении суммаризации статьи №{num_button}, document_id {document_id} {url_part} (статус: {response.status_code})").send()
    except httpx.TimeoutException:
        await cl.Message(content=f"Таймаут при суммаризации статьи  №{num_button}, document_id {document_id} {url_part}").send()
    except Exception as e:
        logger.error(f"Ошибка при запросе суммаризации: {e}")
        await cl.Message(content=f"Ошибка при суммаризации статьи  №{num_button}, document_id {document_id} {url_part}: {str(e)}").send()


if __name__ == "__main__":
    cl.run()