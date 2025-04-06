from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever

class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def _get_relevant_documents(self, query):
        # calculate embeddings for the 'query' string
        emb = self.embeddings.embed_query(query)
        # embeddings = OpenAIEmbeddings() # Kind of hard-coding it to use OpenAI, bad idea
        # emb = embeddings.embed_query(query)


        # take embeddings and feed them into that
        # max_marginal_relevance_search_by_vector

        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8
        )
    
    async def _aget_relevant_documents(self, query):
        return []