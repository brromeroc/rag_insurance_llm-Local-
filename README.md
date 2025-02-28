Bienvenido a rah_insurance_llm, en este repositorio encontrara un agente experto en el documento que se le suministe, esta hecho en python con un modelo local llamado llama3.2 (3B parameters, 2 GB)
el cual puede obtener descargando ollama (https://ollama.com/search) y  luego yendo a su cmd y colocando ollama run llama3.2.

Para instalar las dependencias que utiliza el agente utilice pip install -r requirements.txt.

Para ejecutarlo solomante abra una terminal cmd en la carpeta donde se encuentre rag_insurance_llm.py, y escriba streamlit run rag_insurance_llm.py

Si quiere interacturar con el agente sin instalar nada, puede ingresar al siguiente link: https://gif-improving-discussing-donate.trycloudflare.com/

**NOTA**: En el archivo PDF de la prueba  se uso la métrica SequenceMatcher para saber que tan  bien lo hacia el modelo, dependiendo de que tantos caracteres se repetian en ambas cadenas
(la que se daba y la que el LLM generaba), el problema aparecia cuando la respuesta era demasiado larga o se utlizaban sinonimos, por lo que la métrica daba
numero de presiciónn muy bajos aun cuando las respuestas era acertadas, es por ello que se decidio cambiar la métrica SequenceMatcher por 
BERTscore, que lo que hace es básicamente, ver que tanta relación semantica tienen los embedings de las palabras de referencia y la respuesta del LLM, por medio
de la similitud de coseno (angulo ente lo vectores representativos de las palabras), y esto se promedio tomando como referencia la respuesta de Referencia (P) y luego
se hace tomando de referencia la respuesta del LLM (R), por ultimo se toman estos dos promedios y se hace el ratio de la misma manera que en el archivo. 
Desglosado seria de la siguiente manera: 

Se calcula la presición ($P$) por medio de la similitud de coseno

$$P = \frac{1}{|\hat{x}|} \\sum_{\hat{x}_j\in \hat{x}}\max_{x_i\in x}$$

y el Recall se calcula como:

$$
R = \\frac{1}{|x|} \\sum_{x_i \\in x} \\max_{\\hat{x}_j \\in \\hat{x}} \\text{sim}(x_i, \\hat{x}_j)
$$

Finalmente, el ratio se obtiene combinando ambos:

$$
F1 = \\frac{2 \\times P \\times R}{P + R}
$$
