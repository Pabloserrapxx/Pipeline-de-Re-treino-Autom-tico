🔄 Pipeline de Retreino Automático
Stack: Python, Scikit-Learn, Joblib, Pandas.

Focado na engenharia de Machine Learning clássica, este script automatiza o ciclo de vida de treinamento de modelos, endereçando problemas comuns de reprodutibilidade.

Automação de Processos: O pipeline carrega dados brutos, realiza a divisão estratificada entre treino e teste e treina um classificador Random Forest sem intervenção manual.

Persistência e Versionamento: Após a validação de métricas (Acurácia), o modelo é automaticamente serializado (com joblib) e salvo em um diretório específico, pronto para ser implantado em produção ou comparado com versões anteriores.
