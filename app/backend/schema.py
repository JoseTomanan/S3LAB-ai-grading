from pydantic import BaseModel


class GetItemsRequest(BaseModel):
	...

class GetItemsResponse(GetItemsRequest):
	...