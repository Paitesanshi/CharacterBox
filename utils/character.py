from pydantic import BaseModel

class CharacterInfo(BaseModel):
    id: int
    name: str
    position: str
    states: str
    description: str

    def __str__(self):
        return f'Name: {self.name}\nPosition: {self.position}\nStates: {self.states}\nDescription: {self.description}'

class SceneInfo(BaseModel):
    id: int
    event: str
    time: str
    location: str
    description: str
    plot: str=""

    def __str__(self):
        return f'Event: {self.event}\nTime: {self.time}\nLocation: {self.location}\nDescription: {self.description}'