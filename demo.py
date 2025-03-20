from urielplus import urielplus

u = urielplus.URIELPlus()

u.reset()

#Configuration
u.set_cache(True)

#Integrating databases
u.integrate_databases()

u.set_aggregation('A')

#Imputation
u.softimpute_imputation()

#Distance Calculation
print(u.new_distance("featural", "stan1290", "stan1293"))