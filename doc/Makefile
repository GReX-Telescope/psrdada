LOCAL_ROOT := ../../..
include $(LOCAL_ROOT)/include/Makefile.standard

DOCS := Specification.tex

Specification.dvi: layout.eps intro.tex data_flow.tex data_reduction.tex    \
	command_monitoring.tex control_interface.tex			    \
	configuration_scheduling.tex summary.tex ipc.tex header.tex log.tex \
	socket.tex

include $(LOCAL_ROOT)/include/Makefile.tex

