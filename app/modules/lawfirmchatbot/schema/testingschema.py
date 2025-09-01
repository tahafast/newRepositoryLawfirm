from typing import List, Literal, Optional, Type



class RecommendationPayload(BaseModel):
    userRole: str
    userId: str
    businessId: str
    companyData: str
    recommendationType: str