from AlphaGo.models.policy import CNNPolicy
arch = {'filters_per_layer': 128, 'layers': 12} # args to ResnetPolicy.create_network()
features = ['board','ones','turns_since','liberties','capture_size','self_atari_size','liberties_after','sensibleness','zeros']
policy = CNNPolicy(features, **arch)
policy.save_model('more_filters.json')

