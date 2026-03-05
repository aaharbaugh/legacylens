from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None  # uses settings.query_final_k if unset
    score_threshold: float | None = None  # uses settings if unset
    source_type: str = "all"
    tags: list[str] | None = None  # filter by tags e.g. ["file_io", "para:MAIN"]
    use_reranker: bool = False


class RetrievedChunk(BaseModel):
    id: str
    score: float  # RRF fusion score (for ordering)
    vector_score: float | None = None  # cosine similarity from vector search
    file_path: str
    start_line: int
    end_line: int
    division: str | None
    section_name: str | None
    paragraph_name: str | None
    code_snippet: str | None = None  # omitted in ask/stream responses; fetch via /file-content
    language: str = "COBOL"
    source_type: str = "code"


class QueryResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]


class ChatRequest(BaseModel):
    query: str
    top_k: int | None = None
    source_type: str = "all"
    tags: list[str] | None = None
    use_reranker: bool = False
    chunks: list[RetrievedChunk] | None = None  # when set, skip vector search and use these chunks only


class ChatResponse(BaseModel):
    query: str
    answer: str
    results: list[RetrievedChunk]


class PrefetchRequest(BaseModel):
    query: str
    folder: str = ""


class PrefetchResponse(BaseModel):
    ok: bool = True


class AskRequest(BaseModel):
    query: str
    folder: str = ""


class AskResponse(BaseModel):
    intro: str
    code_snippet: str
    technical_explanation: str
    results: list[RetrievedChunk]


class AdminLoginRequest(BaseModel):
    token: str


class AdminLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AdminReingestRequest(BaseModel):
    code_root: str | None = None
    code_extensions: str | None = None
    batch_size: int | None = None
    max_files: int | None = None
