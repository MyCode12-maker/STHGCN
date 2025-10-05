task_name = 'Test'
city = 'PHO'  # PHO, NYC, SIN
gpuId = "cuda:0"

sample_day_length = 14  # range [3,14]
enable_dynamic_day_length = False
lr = 1e-4
epoch = 10
if city == 'SIN':
    embed_size = 32
    run_times = 3
elif city == 'NYC':
    embed_size = 64
    run_times = 3
elif city == 'PHO':
    embed_size = 64
    run_times = 5

output_file_name = f'{task_name} {city}' + "_epoch" + str(epoch)
output_file_name = output_file_name + '_embeddingSize' + str(embed_size)
