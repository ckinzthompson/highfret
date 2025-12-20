import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

class general_analysis_class(ABC):
	## io
	@abstractmethod
	def save(self) -> None:
		...

	@abstractmethod
	def export(self,format:str, *args, **kwargs) -> None:
		...

	## data - [color,time,nx,ny]
	@property
	@abstractmethod
	def data(self) -> np.ndarray:
		...

	## log - str
	@property
	@abstractmethod
	def log(self) -> str:
		...
	@log.setter
	@abstractmethod
	def log(self, value: str):
		...

	## img -- [color,nx,ny]
	@property
	@abstractmethod
	def img(self) -> np.ndarray:
		...
	@img.setter
	@abstractmethod
	def img(self, value: np.ndarray):
		...

	## transforms -- [from color,to color,theta(K)]
	@property
	@abstractmethod
	def transforms(self) -> np.ndarray:
		...
	@transforms.setter
	@abstractmethod
	def transforms(self, value: np.ndarray):
		...

	## spots -- [color,mol,coordinate(xy)]
	@property
	@abstractmethod
	def spots(self) -> np.ndarray:
		...
	@spots.setter
	@abstractmethod
	def spots(self, value: np.ndarray):
		...

	## traces -- [mol,time,color]
	@property
	@abstractmethod
	def traces(self) -> np.ndarray:
		...
	@traces.setter
	@abstractmethod
	def traces(self, value: np.ndarray):
		...