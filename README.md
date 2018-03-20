# Particle swarm optimization (PSO)
----------

[![Build Status](https://travis-ci.org/yuehhua/PSO.jl.svg?branch=master)](https://travis-ci.org/yuehhua/PSO.jl)
[![codecov.io](http://codecov.io/github/yuehhua/PSO.jl/coverage.svg?branch=master)](http://codecov.io/github/yuehhua/PSO.jl?branch=master)

This is a demo of particle swarm optimization with constraint support in Julia, and this demo is rewrited from [tisimst/pyswarm](https://github.com/tisimst/pyswarm/).

## API

```julia
pso(func, lb, ub; constraints=nothing, args=(), kwargs=Dict(), swarmsize=100,
    omega=0.5, phip=0.5, phig=0.5, maxiter=100, minstep=1e-8, minfunc=1e-8,
    verbose=false, particle_output=false)
```
