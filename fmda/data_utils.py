
def check_data_array(dat,h,a,s):
    try:
        ar = dat[a]
    except:
        print('cannot find array ' + a)
        exit(1)
    print("array %s %s length %i min %s max %s\n" % (a,s,len(ar),min(ar),max(ar)))
    if len(ar) < h:
        print('Error: array length less than %i' % hours)
        exit(1)

def check_data(dat,h2,hours):
    check_data_array(dat,hours,'Ed','drying equilibrium (%)')
    check_data_array(dat,hours,'Ew','wetting equilibrium (%)')
    check_data_array(dat,hours,'rain','rain intensity (mm/h)')
    check_data_array(dat,hours,'fm','RAWS fuel moisture (%)')
