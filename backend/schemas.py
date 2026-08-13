from pydantic import BaseModel, Field

class CreditApplication(BaseModel):
    Age: int
    Sex: str
    Job: int
    Housing: str
    Saving_accounts: str = Field(alias="Saving accounts", default="no_inf")
    Checking_account: str = Field(alias="Checking account", default="no_inf")
    Credit_amount: float = Field(alias="Credit amount")
    Duration: int
    Purpose: str
