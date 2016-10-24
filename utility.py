def mpi_print(form, args, my_rank):
    print '[rank: %d] %s' % (my_rank, form % args )

def root_print(form, args, my_rank):
    if my_rank == 0:
        print '[root] %s' % (form % args)

