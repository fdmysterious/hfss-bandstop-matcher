"""
Touchstone simple file handler for python

Florian Dupeyron
December 2021
"""

import re
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines


from dataclasses import dataclass
from enum        import Enum
from pathlib     import Path


__RE_MODE_LINE=re.compile("#\s*(?P<unit>[A-Za-z]+)\s*(?P<mode>[A-Za-z]+)\s*(?P<repr>[A-Za-z]+)\s*R\s*(?P<ref_ohm>[0-9]+)")
__RE_PORT_COUNT=re.compile("s([0-9]+)p") # File extension is s1p for one port, s2p for 2 port, s3p, etc.

class Touchstone_Unit(Enum):
	Hz  = "Hz"
	kHz = "kHz"
	MHz = "MHz"
	GHz = "GHz"

	@classmethod
	def from_str(cls, svalue):
		values = {
			"hz":  cls.Hz,
			"khz": cls.kHz,
			"mhz": cls.MHz,
			"ghz": cls.GHz
		}

		return values[svalue.lower()]

class Touchstone_Parameter_Mode(Enum):
	Scattering = "S"
	Admittance = "Y"
	Impedance  = "Z"
	Hybrid_H   = "H"
	Hybrid_G   = "G"

	@classmethod
	def from_str(cls, svalue):
		return cls(svalue.upper())

class Touchstone_Repr_Mode(Enum):
	Polar_dB = "DB" # Magnitude [dB] / Angle [deg]
	Polar    = "MA" # Magnitude/Angle [deg]
	Complex  = "RI" # Real/Imaginary

	@classmethod
	def from_str(cls, svalue):
		return cls(svalue.upper())

@dataclass
class Touchstone_Mode_Data:
	unit:    Touchstone_Unit
	params:  Touchstone_Parameter_Mode
	vrepr:   Touchstone_Repr_Mode
	ref_ohm: int

def from_file(path: Path, port_count=None):
	"""
	Loads a touchstone file from the given path.

	:param path: The path to the input file
	:param port_count: The number of ports (1,2,3,4,5,etc.) ; if None, deduced from file name.
	:return: A tuple containing the type of represented values, the reference
	ohm value, the frequencies, and the values as an array.
	"""

	unit_mult = {
		Touchstone_Unit.Hz:  0.0,
		Touchstone_Unit.kHz: 3.0,
		Touchstone_Unit.MHz: 6.0,
		Touchstone_Unit.GHz: 9.0
	}

	mode   = None

	freqs  = []
	values = []

	path = Path(path).resolve()

	# Deduce port count if needed.
	if port_count is None:
		ext = path.name.split(".")[-1] # Get last element of file name.
		mt  = __RE_PORT_COUNT.match(ext)

		if not mt:
			raise ValueError("Cannot deduce port count from file name: invalid file extension")

		port_count = int(mt.group(1)) # First group of the regex contains the port count


	# Open and process file
	with open(str(path), "r") as fhandle:
		for line in fhandle:
			line = line.strip()

			# if line is empty, strip will return an empty string,
			# so will not enter in condition :D
			if line:
				if   line[0] == "!": # Comment line
					continue
				elif line[0] == "#": # Mode line!
					line_data = __RE_MODE_LINE.match(line)
					if not line_data:
						raise ValueError(f"Invalid mode line: {line!r}")

					# Extract infos
					mode = Touchstone_Mode_Data(
						unit    = Touchstone_Unit.from_str(line_data.group("unit")),
						params  = Touchstone_Parameter_Mode.from_str(line_data.group("mode")),
						vrepr   = Touchstone_Repr_Mode.from_str(line_data.group("repr")),
						ref_ohm = int(line_data.group("ref_ohm"))
					)

				else:
					if mode is None:
						raise ValueError("Values are given without mode information!")

					# Parse line data
					# -> split line by spaces chars, and remove empty elements to keep only needed columns
					line_data = filter(lambda x: len(x) > 0, map(lambda x: x.strip(), line.split(" ")))

					# Parse frequency, multiply by unit
					freq   = float(next(line_data))*pow(10, unit_mult[mode.unit])
					vports = [] # Values for current row

					# Parse values
					for i in range(port_count*2):
						v1 = float(next(line_data))
						v2 = float(next(line_data))

						# put in array, convert all values to complex numbers
						#vports.append((v1,v2))
						if   mode.vrepr == Touchstone_Repr_Mode.Complex:
							vports.append( v1+1j*v2 )

						elif mode.vrepr == Touchstone_Repr_Mode.Polar:
 							vports.append( v1*np.exp(1j*np.deg2rad(v2)) )

						elif mode.vrepr == Touchstone_Repr_Mode.Polar_dB:
							amp = 10.0 ** (v1/20.0)
							vports.append(amp*np.exp(1j*np.deg2rad(v2)))


					if vports:
						freqs.append(freq)
						values.append(vports)

	# Return results
	return (mode.params, mode.ref_ohm, freqs, values)


# TODO
#def s21_min_plot(label, ax, linestyle="solid", color=""):

def s21_min(fpath):
	mode, ref_ohm, freqs, values = from_file(fpath)

	freqs     = np.array(freqs)/1e9
	
	s21        = np.array(values)[:,1]
	s21_dB     = 20*np.log10(np.abs(s21))
	
	print(s21_dB.shape)
	
	#s21_dB    = np.array(values)[:,1,0]
	#s21_dB = 20*np.log(np.abs(s21))
	
	# Limit axes for research to 2GHz
	i_end = np.where(freqs >= 2.0)[0][0]
	
	# Find min
	v_min = np.min(s21_dB[:i_end])
	i_min = np.where(s21_dB == v_min)[0][0]
	f_min = freqs[i_min]

	return f_min, v_min, freqs, s21_dB