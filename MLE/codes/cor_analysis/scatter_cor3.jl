left_margin=50px

using CSV
using DelimitedFiles
using DataFrames
using HDF5
using Plots
using Dates
using StatsBase
using LaTeXStrings
using LinearAlgebra
using Polynomials
using Optim
using Distributions
using LsqFit
using DifferentialEquations
#using StatsPlots
using Measures
Pkg.update()


# in utc:
#=Tfid=h5open("../data/other/CE_data_collected_by_johannes/input_actual_timezone_utc.h5","r")
Tdgr=Tfid["df"]
#println(names(dgr))
Tdata=read(Tdgr["block0_values"])
Tsces=read(Tdgr["block0_items"])
Tsce1=read(Tdgr["axis0"])
Tsce2=read(Tdgr["axis1"])
Tsce2=unix2datetime.(Tsce2[:]/1000000000)=#

#in timezone berlin:
fid=h5open("../data/other/CE_data_collected_by_johannes/input_actual_timezone_berlin.h5","r")
dgr=fid["df"]
#println(names(dgr))
data=read(dgr["block0_values"])
sces=read(dgr["block0_items"])
sce1=read(dgr["axis0"])
sce2=read(dgr["axis1"])
sce2=unix2datetime.(sce2[:]/1000000000)
#sce2[35065+0:(35065+364*24)]
sce2[26305+0:(26305+364*24+23)]
startd=26305

# 1) use goods2 from model_atPeak/parameter_check_mle

# 2)now filter the intervals that ar both in goods2 as well as complete in Johannes' dataset
cort=zeros(length(goods2))
nanent=zeros(length(goods2))
j=1
nj=1
for i in goods2
    if maximum(isnan.(data[:,startd+i]))==false #iff no Nan entry in this column
        cort[j]=i
        global j+=1
    else
        nanent[nj]=i
        global nj+=1
    end
end
cor=convert.(Int64,cort[1:j-1]) #alle, die rausfliegen
nanen=convert.(Int64,nanent[1:nj-1]) # alle intervalle die gehen

# calculate the correlations
spearman=zeros(49,5)
for jo in 1:49
    for mle in 1:5
        spearman[jo,mle]=corspearman(data[jo,cor.+startd],mleres[cor,2+mle])
    end
end
spearman

# make the heatmap
scespl=["\$\\textrm{load}\$",
    "\$\\textrm{gen \\ biomas}\$",
    "\$\\textrm{gen \\ lignite}\$",
    "\$\\textrm{gen \\ coal-gas}\$",
    "\$\\textrm{gen \\ gas}\$",
    "\$\\textrm{gen \\ hardcoal}\$",
    "\$\\textrm{gen \\ oil}\$",
    "\$\\textrm{gen \\ geothermal}\$",
    "\$\\textrm{gen \\ pumped \\ hydro}\$",
    "\$\\textrm{gen \\ run \\ off \\ hydro}\$",
    "\$\\textrm{gen \\ reservoir \\ hydro}\$",
    "\$\\textrm{gen \\ nuclear}\$",
    "\$\\textrm{gen \\ other \\ renewables}\$",
    "\$\\textrm{gen \\ solar}\$",
    "\$\\textrm{gen \\ waste}\$",
    "\$\\textrm{gen \\ wind \\ off}\$",
    "\$\\textrm{gen \\ wind \\ on}\$",
    "\$\\textrm{gen \\ other}\$",
    "\$\\textrm{gen \\ total}\$",
    "\$\\textrm{gen \\ synchr.}\$",
    "\$\\textrm{load \\ ramp}\$",
    "\$\\textrm{total \\ gen \\ ramp}\$",
    "\$\\textrm{biomass \\ ramp}\$",
    "\$\\textrm{lignite \\ ramp}\$",
    "\$\\textrm{coal \\ gas \\ ramp}\$",
    "\$\\textrm{gas \\ ramp}\$",
    "\$\\textrm{hard \\ coal \\ ramp}\$",
    "\$\\textrm{oil \\ ramp}\$",
    "\$\\textrm{geothermal \\ ramp}\$",
    "\$\\textrm{pumped \\ hydro \\ ramp}\$",
    "\$\\textrm{run \\ off \\ hydro \\ ramp}\$",
    "\$\\textrm{reservoir \\ hydro \\ ramp}\$",
    "\$\\textrm{nuclear \\ ramp}\$",
    "\$\\textrm{other \\ renewables \\ ramp}\$",
    "\$\\textrm{solar \\ ramp}\$",
    "\$\\textrm{waste \\ ramp}\$",
    "\$\\textrm{wind \\ off \\ ramp}\$",
    "\$\\textrm{wind \\ on \\ ramp}\$",
    "\$\\textrm{other \\ ramp}\$",
    "\$\\textrm{fe \\ wind \\ on}\$",
    "\$\\textrm{fe \\ wind \\ off}\$",
    "\$\\textrm{fe \\ solar}\$",
    "\$\\textrm{fe \\ total \\ gen}\$",
    "\$\\textrm{fe \\ load}\$",
    "\$\\textrm{fe \\ load \\ ramp}\$",
    "\$\\textrm{fe \\ total \\ gen \\ ramp}\$",
    "\$\\textrm{fe \\ wind \\ off \\ ramp}\$",
    "\$\\textrm{fe \\ wind \\ on \\ ramp}\$",
    "\$\\textrm{fe \\ solar \\ ramp}\$"]

Plots.heatmap(["\$\\gamma_1\$","\$\\gamma_2\$","\$q\$","\$r\$", "\$\\textrm{D}\$"],
    scespl,
    spearman,
    ticks=:all,
    left_margin=45pt,
    c= :diverging_bwr_40_95_c42_n256,
    size=(800,1200)
)
png("/Users/raphaelbiertz/Documents/masterarbeit/coding/codes/cor_analysis/spearman3")


writedlm("C:/Users/Raphael Biertz/Desktop/spearman.csv",spearman)
writedlm("C:/Users/Raphael Biertz/Desktop/spear.csv",sces)
