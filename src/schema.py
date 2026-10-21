from pydantic import BaseModel, Field, field_validator

ALTERNATIVAS = 4


class Questao(BaseModel):
    enunciado: str = Field(min_length=10)
    alternativas: list[str] = Field(min_length=ALTERNATIVAS, max_length=ALTERNATIVAS)
    correta: int = Field(ge=0, le=ALTERNATIVAS - 1)
    justificativa: str = Field(min_length=10)
    evidencia: str = Field(min_length=5)
    pagina: int = Field(ge=1)

    @field_validator("enunciado", "justificativa", "evidencia")
    @classmethod
    def nao_vazio(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("campo vazio")
        return v

    @field_validator("alternativas")
    @classmethod
    def quatro_distintas(cls, v: list[str]) -> list[str]:
        v = [a.strip() for a in v]
        if len(v) != ALTERNATIVAS:
            raise ValueError(f"esperado {ALTERNATIVAS} alternativas, veio {len(v)}")
        if any(not a for a in v):
            raise ValueError("alternativa vazia")
        if len(set(v)) != ALTERNATIVAS:
            raise ValueError("alternativas repetidas")
        return v

    @property
    def letra_correta(self) -> str:
        return "ABCD"[self.correta]
