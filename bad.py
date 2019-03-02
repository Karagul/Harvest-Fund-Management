import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.utils.data
import pickle
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

raw_data = pd.read_csv('../data/gpd.csv',index_col=2,na_values='..')
raw_data = raw_data.iloc[:,3:]
raw_data = raw_data.dropna(axis=0,how='all').dropna(axis=1,how='all')
raw_data = raw_data.rename({old: int(old.split()[0]) for old in raw_data.columns}, axis=1)
raw_data = raw_data.sort_values(2017,ascending=False).iloc[:80]

percentage_data = raw_data/np.sum(raw_data,axis=0)



rate_data = pd.read_csv('../data/gpd_grow_rate.csv',index_col=0,na_values='..')
rate_data = rate_data.iloc[:,3:]
rate_data = rate_data.dropna(axis=0,how='all').dropna(axis=1,how='all')
rate_data = rate_data.rename({old: int(old.split()[0]) for old in rate_data.columns}, axis=1)
rate_data = rate_data/100




target_years = np.arange(1975,2020,5)
result = []



for country in percentage_data.index:
    country_percentage_data = percentage_data.loc[country]
    country_rate_data = rate_data.loc[country]
    for target_year in target_years:
        target_year_list = [year for year in np.arange(target_year-5,target_year+1)]
        target_percentage_data = country_percentage_data[target_year_list]
        target_rate_data = country_rate_data[target_year_list]
        
        if target_rate_data.isnull().sum() > 0 or target_percentage_data.isnull().sum() > 0:
            continue
        else:
            result.append([country+' '+str(target_year-5)+'~'+str(target_year)]+\
                          target_rate_data.values.tolist()+ target_percentage_data.values.tolist())
            
            
            
final_df = pd.DataFrame(result)


class GDP_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, df, device):
        
        self.df = df
        self.device = device
        self.country_year = self.df.iloc[:,0]
        self.tensor_data = torch.from_numpy(self.df.iloc[:,1:].values.astype('float')).float().to(self.device)
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        return self.country_year[idx],self.tensor_data[idx,:]


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
            



gdp_dataset = GDP_Dataset(final_df, device)
gdp_batch = torch.utils.data.DataLoader(gdp_dataset, batch_size=64, shuffle=True)

batch_iterator =iter(cycle(gdp_batch))


class Generator(torch.nn.Module):

    def __init__(self, target_size, hidden_size=32, z_norm_size=3, z_one_hot_size=4, num_layers=2):
        
        super().__init__()
        self.first_layer = torch.nn.Linear(in_features=z_norm_size+z_one_hot_size, out_features=hidden_size)
        self.first_layer_bn = torch.nn.BatchNorm1d(hidden_size)
        self.last_layer = torch.nn.Linear(in_features=hidden_size, out_features=target_size)
        
        #self.hidden_activation = torch.nn.LeakyReLU(0.2)
        self.hidden_activation = torch.nn.ReLU()
        
        
        self.middle_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size,hidden_size), torch.nn.BatchNorm1d(hidden_size),
                                                  self.hidden_activation])
        
        for i in range(0,num_layers-1):
            self.middle_layers.extend([torch.nn.Linear(hidden_size,hidden_size), torch.nn.BatchNorm1d(hidden_size),
                                                  self.hidden_activation])
            
    def forward(self, x):
        
        x = self.first_layer(x)
        x = self.first_layer_bn(x)
        x = self.hidden_activation(x)
        for layer in self.middle_layers:
            x = layer(x)
        x = self.last_layer(x)
        return x
        
        
        
class Discriminator(torch.nn.Module):
    
    def __init__(self, target_size, hidden_size=32, num_layers=2):
        super().__init__()
        
        self.first_layer = torch.nn.Linear(in_features=target_size, out_features=hidden_size)
        self.first_layer_bn = torch.nn.BatchNorm1d(hidden_size)
        self.last_layer = torch.nn.Linear(in_features=hidden_size, out_features=1)
        
        #self.hidden_activation = torch.nn.LeakyReLU(0.2)
        self.hidden_activation = torch.nn.ReLU()
        
        self.middle_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size,hidden_size), torch.nn.BatchNorm1d(hidden_size),
                                                  self.hidden_activation])
        
        for i in range(0,num_layers-1):
            self.middle_layers.extend([torch.nn.Linear(hidden_size,hidden_size), torch.nn.BatchNorm1d(hidden_size),
                                                  self.hidden_activation])
            
            
    def forward(self,x):
        
        x = self.first_layer(x)
        x = self.first_layer_bn(x)
        x = self.hidden_activation(x)
        for layer in self.middle_layers:
            x = layer(x)
        x = self.last_layer(x)
        
        return x.flatten()
        
        
        
        
class Encoder(torch.nn.Module):
    
    def __init__(self, target_size, hidden_size=32, z_norm_size=3, z_one_hot_size=4, num_layers=2):
        super().__init__()
        
        self.z_norm_size = z_norm_size
        self.z_one_hot_size = z_one_hot_size
        
        self.first_layer = torch.nn.Linear(in_features=target_size, out_features=hidden_size)
        self.first_layer_bn = torch.nn.BatchNorm1d(hidden_size)
        
        self.last_layer = torch.nn.Linear(in_features=hidden_size, out_features=z_norm_size+z_one_hot_size)
        
        #self.hidden_activation = torch.nn.LeakyReLU(0.2)
        self.hidden_activation = torch.nn.ReLU()
        
        
        self.middle_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size,hidden_size), torch.nn.BatchNorm1d(hidden_size),
                                                  self.hidden_activation])
        
        for i in range(0,num_layers-1):
            self.middle_layers.extend([torch.nn.Linear(hidden_size,hidden_size), torch.nn.BatchNorm1d(hidden_size),
                                                  self.hidden_activation])
            
    def forward(self, x):
        
        x = self.first_layer(x)
        x = self.first_layer_bn(x)
        x = self.hidden_activation(x)
        for layer in self.middle_layers:
            x = layer(x)
        x = self.last_layer(x)
        
        
        return x[:,:self.z_norm_size], x[:,-self.z_one_hot_size:]
    
    
    
    
def generate_Models(target_size, z_norm_size=3, z_one_hot_size=4, hidden_size=16, num_layers=2,
                   device = device):
    generator = Generator(target_size, hidden_size=hidden_size, z_norm_size=z_norm_size,
                          z_one_hot_size=z_one_hot_size, num_layers=num_layers).to(device)
    discriminator = Discriminator(target_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    encoder = Encoder(target_size, hidden_size=hidden_size, z_norm_size=z_norm_size, 
                      z_one_hot_size=z_one_hot_size, num_layers=num_layers).to(device)
    
    
    return generator, discriminator, encoder

def get_Optimizer(generator, discriminator, encoder, lr=0.00005, weight_decay=0.001):
    generator_opti = torch.optim.RMSprop([para for para in generator.parameters() if para.requires_grad],
                                         lr=lr,weight_decay=weight_decay)
    discriminator_opti = torch.optim.RMSprop([para for para in discriminator.parameters() if para.requires_grad],
                                             lr=lr, weight_decay=weight_decay)
    encoder_opti = torch.optim.RMSprop([para for para in encoder.parameters() if para.requires_grad],lr=lr, 
                                       weight_decay=weight_decay)
    return generator_opti, discriminator_opti, encoder_opti
    
    
def para_init(model):
    for name, para in model.named_parameters():
        if len(para.shape) > 1:
            torch.nn.init.normal_(para, mean=0, std=0.01)
        
        
  
  
  
def validation(generator, z_norm_size=3, z_one_hot_size=4, weight_decay=0.0, device=device):
    total_loss = 0
    generator.eval()
    for batch_label, batch_data in gdp_batch_val:
        record = None
        for k in range(z_one_hot_size):
            batch_size = batch_data.shape[0]
            z_n = torch.zeros(batch_size, z_norm_size).normal_(0, 0.1).to(device)
            z_n.requires_grad=True
            z_c = torch.zeros(batch_size, z_one_hot_size).to(device)
            z_c[:,k] = 1
            z = torch.cat([z_n, z_c], dim=1)

            opti = torch.optim.Adam([z_n], lr=0.001, weight_decay=weight_decay)

            for time in range(1000):
                opti.zero_grad()
                loss = torch.nn.functional.mse_loss(generator(z), batch_data.float())
                loss.backward()
                opti.step()
                z_n.data.clamp_(-0.6, 0.6)
                z = torch.cat([z_n, z_c], dim=1)

            k_loss = torch.sum(torch.nn.functional.mse_loss(generator(z), batch_data.float(),reduction='none'), dim=1).view(-1,1)
            if record is None:
                record = k_loss.detach()
            else:
                record = torch.cat([record,k_loss.detach()], dim=1)

        batch_min, batch_index = record.min(1)
        total_loss += torch.sum(batch_min).item()
        
    return total_loss



def encoder_generator(generator, encoder, z_norm_size=3, z_one_hot_size=4, weight_decay=0.0, device=device):
    total_loss = 0
    generator.eval()
    encoder.eval()
    for batch_label, batch_data in gdp_batch_val:
        batch_data = batch_data.float()
        z_n,z_c = encoder(batch_data)
        batch_size = batch_data.shape[0]
        nb_digits = z_c.shape[1]
        v,y = z_c.max(dim=1)
        y = y.view(-1,1)
        y_onehot = torch.FloatTensor(batch_size, nb_digits).to(device)

        
        y_onehot.zero_()

        y_onehot.scatter_(1, y, 1)
        z = torch.cat([z_n, y_onehot], dim=1)
        loss = torch.nn.functional.mse_loss(generator(z), batch_data, reduction='sum') 
        total_loss += loss.detach().item()
        
    return total_loss
            
            
def train(batch_data, generator, discriminator, encoder, generator_opti, discriminator_opti, encoder_opti,
         c = 0.02, n_critic=5, z_norm_size=3, z_one_hot_size=4, default_batch_size=64, total_times = 10000):
    current_time = 0
    record1_loss = []
    record2_loss = []
    val_loss = []
    e_g_loss = []
    
    best_loss = float('inf')
    
    
    while current_time < total_times:
        current_time += 1
        generator.train()
        discriminator.train() 
        encoder.train() 
        
        for times in range(n_critic):
            
            discriminator_opti.zero_grad()
            
            real_label, real_data = next(batch_data)
            real_data = real_data.float()
            batch_size = real_data.shape[0]
        
            z_n = torch.zeros(batch_size, z_norm_size).normal_(0, 0.1).to(device)
            idx = torch.randint(0,z_one_hot_size,size=(1,batch_size)).squeeze()
            z_c = torch.zeros(len(idx), idx.max()+1).scatter_(1, idx.unsqueeze(1), 1.).to(device)
            
            if z_c.shape[1] < z_one_hot_size:
                z_c = torch.cat([z_c, torch.zeros(batch_size, z_one_hot_size - z_c.shape[1]).to(device)], dim=1)
            
            z = torch.cat([z_n,z_c], dim=1)
            
            generated_data = generator(z)
            
            real_score = discriminator(real_data)
            
            fake_score = discriminator(generated_data)
            
            #loss = -(torch.mean(real_score) - torch.mean(fake_score))
            loss = -(torch.log(torch.nn.functional.sigmoid(real_score)).mean() +\
                    torch.log(1-torch.nn.functional.sigmoid(fake_score)).mean())
            
            
            loss.backward()
            
            discriminator_opti.step()
            
                    
            record1_loss.append(loss.detach().item())
                    
        
        encoder_opti.zero_grad()
        generator_opti.zero_grad()
        
        z_n = torch.zeros(default_batch_size, z_norm_size).normal_(0, 0.1).to(device)
        idx = torch.randint(0,z_one_hot_size,size=(1,default_batch_size)).squeeze()
        z_c = torch.zeros(len(idx), idx.max()+1).scatter_(1, idx.unsqueeze(1), 1.).to(device)
        idx = idx.to(device)
        
        if z_c.shape[1] < z_one_hot_size:
                z_c = torch.cat([z_c, torch.zeros(default_batch_size, z_one_hot_size - z_c.shape[1]).to(device)], dim=1)
                
        z = torch.cat([z_n,z_c], dim=1).to(device)
        
        generated_data = generator(z)
        fake_score = discriminator(generated_data)
        z_n_fake, z_c_fake = encoder(generated_data)
        
        
        generator_loss = -torch.log(torch.nn.functional.sigmoid(fake_score)).mean() +\
                            torch.nn.functional.mse_loss(z_n, z_n_fake) +\
                            torch.nn.functional.cross_entropy(z_c_fake, idx)
                
                
        generator_loss.backward()
        encoder_opti.step()
        generator_opti.step()
        
        
        record2_loss.append(generator_loss.detach().item())
        
        
        if current_time == 1 or current_time%200 == 0:
            temp_e_g_loss = encoder_generator(generator, encoder, weight_decay=back_weight_decay,z_norm_size=z_norm_size, 
                    z_one_hot_size=z_one_hot_size, device=device)
            e_g_loss.append(temp_e_g_loss)
            
            temp_loss = validation(generator, weight_decay=back_weight_decay,z_norm_size=z_norm_size, 
                    z_one_hot_size=z_one_hot_size, device=device)
            print(temp_loss)
            sys.stdout.flush()
            val_loss.append(temp_loss)
            
            if temp_loss < best_loss:
                best_loss = temp_loss
                torch.save(generator.state_dict(), '../store/'+'bad'+'/generator.pt')
                torch.save(discriminator.state_dict(), '../store/'+'bad'+'/discriminator.pt')
                torch.save(encoder.state_dict(), '../store/'+'bad'+'/encoder.pt')
        
    return record1_loss, record2_loss, val_loss, e_g_loss


gde_weight_decay = 0.001
back_weight_decay = 0.1
z_norm_size=2
z_one_hot_size=4
target_size = 12
hidden_size = 32



generator, discriminator, encoder = generate_Models(target_size,z_norm_size=z_norm_size,z_one_hot_size=z_one_hot_size,
                                                   hidden_size = hidden_size)
generator_opti, discriminator_opti, encoder_opti = get_Optimizer(generator, discriminator, encoder, 
                                                                 weight_decay=gde_weight_decay)



gdp_batch_val = torch.utils.data.DataLoader(gdp_dataset, batch_size=64, shuffle=True)

record1_loss, record2_loss, val_loss, e_g_loss = train(cycle(gdp_batch), generator, discriminator, encoder, 
                                   generator_opti, discriminator_opti, encoder_opti,z_norm_size=z_norm_size,
                                             z_one_hot_size=z_one_hot_size)


pickle.dump(record1_loss, open('../store/'+'bad'+'/record1_loss.pt','wb'))
pickle.dump(record2_loss, open('../store/'+'bad'+'/record2_loss.pt','wb'))
pickle.dump(val_loss, open('../store/'+'bad'+'/val_loss.pt','wb'))
pickle.dump(e_g_loss, open('../store/'+'bad'+'/e_g_loss.pt','wb'))

torch.save(generator.state_dict(), '../store/'+'bad'+'/generator_final.pt')
torch.save(discriminator.state_dict(), '../store/'+'bad'+'/discriminator_final.pt')
torch.save(encoder.state_dict(), '../store/'+'bad'+'/encoder_final.pt')


def final_val(generator, z_norm_size=3, z_one_hot_size=4, weight_decay=0.0, device=device):
    total_loss = 0
    generator.eval()
    final_result = []
    
    for batch_label, batch_data in gdp_batch_val:
        record = None
        for k in range(z_one_hot_size):
            batch_size = batch_data.shape[0]
            z_n = torch.zeros(batch_size, z_norm_size).normal_(0, 0.1).to(device)
            z_n.requires_grad=True
            z_c = torch.zeros(batch_size, z_one_hot_size).to(device)
            z_c[:,k] = 1
            z = torch.cat([z_n, z_c], dim=1)

            opti = torch.optim.Adam([z_n], lr=0.001, weight_decay=weight_decay)

            for time in range(1000):
                opti.zero_grad()
                loss = torch.nn.functional.mse_loss(generator(z), batch_data.float())
                loss.backward()
                opti.step()
                z_n.data.clamp_(-0.6, 0.6)
                z = torch.cat([z_n, z_c], dim=1)

            k_loss = torch.sum(torch.nn.functional.mse_loss(generator(z), batch_data.float(),reduction='none'), dim=1).view(-1,1)
            if record is None:
                record = k_loss.detach()
            else:
                record = torch.cat([record,k_loss.detach()], dim=1)

        batch_min, batch_index = record.min(1)
        t = [(' '.join(country.split()[:-1]), country.split()[-1],group.item()) for country,group in zip(batch_label, batch_index)]
        final_result.extend(t)
        total_loss += torch.sum(batch_min).item()
        
    return final_result
            
final_result = final_val(generator, weight_decay=back_weight_decay,z_norm_size=z_norm_size,
                                             z_one_hot_size=z_one_hot_size, device=device)

final_df = pd.DataFrame(final_result)

pickle.dump(final_df, open('../store/'+'bad'+'/final_df.pt','wb'))
