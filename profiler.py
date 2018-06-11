import pstats

file_python = 'standard_python'
file_cython = 'c_python'
p = pstats.Stats(file_python)
p.print_stats()

p = pstats.Stats(file_cython)
p.print_stats()
