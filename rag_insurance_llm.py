import streamlit as st
import PyPDF2
import difflib
import logging
import time
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app_logs.log',
    filemode='a'
)

# Función que extrae el texto del documento
def extraer_texto_pdf_paginas(pdf_file, paginas):
    texto = ""
    lector = PyPDF2.PdfReader(pdf_file)
    total_paginas = len(lector.pages)
    for num in paginas:
        if 0 <= num < total_paginas:
            pagina = lector.pages[num]
            texto += pagina.extract_text() + "\n"
        else:
            st.error(f"La página {num+1} no existe en el documento.")
            logging.warning(f"Intento de acceso a página inexistente: {num+1}")
    return texto

def main():
    st.title("Análisis de PDF con Langchain, Ollama y Streamlit")
    pdf_file = st.file_uploader("Carga tu archivo PDF", type="pdf")

    if pdf_file is not None:
        lector = PyPDF2.PdfReader(pdf_file)
        total_paginas = len(lector.pages)
        st.write(f"El PDF tiene **{total_paginas}** páginas.")

        paginas_input = st.text_input(
            "Ingresa los números de las páginas que deseas analizar (máximo 4, separados por comas):",
            "1,2,3,4"
        )
        try:
            paginas = [int(x.strip()) - 1 for x in paginas_input.split(",") if x.strip().isdigit()]
            if not paginas:
                st.error("No se ingresaron números de página válidos.")
                return
            paginas = paginas[:4]
        except Exception as e:
            st.error(f"Error al procesar las páginas: {e}")
            logging.error(f"Error al procesar las páginas: {e}")
            return

        pdf_file.seek(0)
        texto_seleccionado = extraer_texto_pdf_paginas(pdf_file, paginas)
        if not texto_seleccionado:
            st.error("No se pudo extraer texto de las páginas seleccionadas.")
            return
        st.success("Se han extraído las páginas seleccionadas correctamente.")

        pregunta = st.text_area("Ingresa tu pregunta o instrucción sobre el documento", "")
        ground_truth = st.text_area("Ingresa la respuesta esperada (opcional, para evaluar la precisión)", "")

        if st.button("Procesar consulta"):
            if not pregunta:
                st.error("Por favor, ingresa una pregunta o instrucción.")
            else:
                st.info("Procesando la consulta...")
                logging.info("Procesando consulta con el modelo.")

                llm = Ollama(model="llama3.2")

                prompt_template = PromptTemplate(
                    input_variables=["contexto", "pregunta"],
                    template="""
                    Usa el siguiente contexto para responder la pregunta:
                    Contexto:
                    {contexto}

                    Pregunta:
                    {pregunta}
                    """
                )

                chain = LLMChain(llm=llm, prompt=prompt_template)

                #tiempo de ejecución
                start_time = time.time()
                respuesta = chain.run({"contexto": texto_seleccionado, "pregunta": pregunta})
                end_time = time.time()
                latencia = end_time - start_time

                st.write("### Respuesta:")
                st.write(respuesta)

                st.write(f"**Latencia del modelo:** {latencia:.2f} segundos")
                logging.info(f"Latencia del modelo: {latencia:.2f} segundos")

                if ground_truth:
                    similitud = difflib.SequenceMatcher(None, respuesta, ground_truth).ratio()
                    st.write(f"**Precisión estimada (similitud): {similitud*100:.2f}%**")
                    logging.info(f"Precisión estimada: {similitud*100:.2f}%")

if __name__ == "__main__":
    main()
