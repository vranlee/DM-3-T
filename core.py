# ------------------------------------------------------------------------
# Supplementary Material - Core Implementation Source Code
# DM^3T: Harmonizing Modalities via Diffusion for Multi-Object Tracking
# ------------------------------------------------------------------------

class CrossModalDiffusionFusion(nn.Module):
    def __init__(self, channels, steps=3, noise_level=0.01):
        super(CrossModalDiffusionFusion, self).__init__()
        self.steps = steps
        self.noise_level = noise_level

        self.denoise_rgb = self._make_denoise_net(channels * 2, channels)
        self.denoise_t = self._make_denoise_net(channels * 2, channels)

    def _make_denoise_net(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
        )

    def forward(self, x_rgb, x_t):
        refined_rgb = x_rgb
        refined_t = x_t

        for _ in range(self.steps):
            noisy_rgb = refined_rgb + torch.randn_like(refined_rgb) * self.noise_level
            noisy_t = refined_t + torch.randn_like(refined_t) * self.noise_level

            guidance_for_rgb = torch.cat([noisy_rgb, refined_t], dim=1)
            residual_rgb = self.denoise_rgb(guidance_for_rgb)
            refined_rgb = noisy_rgb + residual_rgb

            guidance_for_t = torch.cat([noisy_t, refined_rgb], dim=1)
            residual_t = self.denoise_t(guidance_for_t)
            refined_t = noisy_t + residual_t

        return refined_rgb + refined_t

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class DiffusionModule(nn.Module):
    def __init__(self, channels, steps=3, noise_level=0.1, refinement_strength=0.2):
        super(DiffusionModule, self).__init__()
        self.steps = steps
        self.noise_level = noise_level
        self.refinement_strength = refinement_strength

        self.time_embed_dim = 32
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.noise_predictor = nn.Sequential(
            nn.Conv2d(channels, channels//2, kernel_size=1),
            nn.BatchNorm2d(channels//2, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.denoise_layers = nn.ModuleList([
            nn.ModuleDict({
                'main': nn.Sequential(
                    nn.Conv2d(channels + self.time_embed_dim, channels*2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels*2, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels*2, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                ),
                'attn': nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=1),
                    nn.Sigmoid()
                )
            }) for _ in range(steps)
        ])

        self.skip_connections = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1)
            for _ in range(steps)
        ])

        self.output_layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

    def _add_time_embedding(self, x, t):
        t_emb = self.time_embed(t.reshape(-1, 1).float())
        t_emb = t_emb.reshape(-1, self.time_embed_dim, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        return torch.cat([x, t_emb], dim=1)

    def forward(self, x):
        x_original = x.clone()
        batch_size = x.shape[0]

        noise_scale = self.noise_predictor(x)

        x_noisy = x + self.noise_level * noise_scale * torch.randn_like(x)

        for i in range(self.steps):
            t = torch.ones(batch_size, device=x.device) * (1.0 - i / self.steps)

            x_with_time = self._add_time_embedding(x_noisy, t)

            denoise_out = self.denoise_layers[i]['main'](x_with_time)

            attention_map = self.denoise_layers[i]['attn'](x_noisy)
            denoise_out = denoise_out * attention_map

            residual = self.skip_connections[i](x_noisy)
            x_noisy = denoise_out + residual

        x_refined = self.output_layer(x_noisy)

        enhanced = x_original + self.refinement_strength * (x_refined - x_original)

        return enhanced

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False,
                 opt=None):
        super(DLA, self).__init__()

        
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)
        if opt.pre_img:
            self.pre_img_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        if opt.pre_hm:
            self.pre_hm_layer = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=7, stride=1,
                    padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))

        self.diffusion_fusion = CrossModalDiffusionFusion(
            channels=channels[0],
            steps=opt.diffusion_steps if hasattr(opt, 'diffusion_steps') else 3,
            noise_level=opt.noise_level if hasattr(opt, 'noise_level') else 0.01
        )
        print(f"INFO: Using Cross-Modal Diffusion Fusion with {self.diffusion_fusion.steps} steps.")

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)
    def token2feature(self,tokens,h,w):
        L,B,D=tokens.shape
        H,W=h,w
        x = tokens.permute(1, 2, 0).view(B, D, H, W).contiguous()
        return x
    
    def feature2token(self,x):
        B,C,W,H = x.shape
        L = W*H
        tokens = x.view(B, C, L).permute(2, 0, 1).contiguous()
        return tokens
    def get_positional_encoding(self, feat):
        b, _, h, w = feat.shape

        mask = torch.zeros((b, h, w), dtype=torch.bool, device=feat.device)
        pos = self.pos_encoding(mask)
        return pos.reshape(b, -1, h, w)

    def forward(self, x_rgb, x_t, pre_img_rgb=None, pre_img_t=None, pre_hm=None):
        y = []
        x_rgb = self.base_layer(x_rgb)
        x_t = self.base_layer(x_t)

        if pre_img_rgb is not None:
            x_rgb = x_rgb + self.pre_img_layer(pre_img_rgb)
        if pre_img_t is not None:
            x_t = x_t + self.pre_img_layer(pre_img_t)
        if pre_hm is not None:
            pre_hm_feat = self.pre_hm_layer(pre_hm)
            x_rgb = x_rgb + pre_hm_feat
            x_t = x_t + pre_hm_feat

        x = self.diffusion_fusion(x_rgb, x_t)

        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)

        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights, strict=False)


class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f, node_type=(DeformConv, DeformConv)):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  
            proj = node_type[0](c, o)
            node = node_type[1](o, o)
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])



class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None, 
                 node_type=DeformConv):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j],
                          node_type=node_type))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class DLASeg(BaseModel):
    def __init__(self, num_layers, heads, head_convs, opt):
        super(DLASeg, self).__init__(
            heads, head_convs, 1, 64 if num_layers == 34 else 128, opt=opt)
        
        down_ratio=4
        self.opt = opt
        self.node_type = DLA_NODE[opt.dla_node]
        print('Using node type:', self.node_type)
        self.first_level = int(np.log2(down_ratio))
        self.last_level = 5
        self.base = globals()['dla{}'.format(num_layers)](
            pretrained=(opt.load_model == ''), opt=opt)

        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(
            self.first_level, channels[self.first_level:], scales,
            node_type=self.node_type)
        out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
            out_channel, channels[self.first_level:self.last_level], 
            [2 ** i for i in range(self.last_level - self.first_level)],
            node_type=self.node_type)
        
        self.use_diffusion = opt.use_diffusion if hasattr(opt, 'use_diffusion') else False
        if self.use_diffusion:
            self.diffusion = DiffusionModule(
                channels=out_channel,
                steps=opt.diffusion_steps if hasattr(opt, 'diffusion_steps') else 3,
                noise_level=opt.noise_level if hasattr(opt, 'noise_level') else 0.1,
                refinement_strength=opt.refinement_strength if hasattr(opt, 'refinement_strength') else 0.2
            )
            print('Using Diffusion Module with {} steps'.format(
                opt.diffusion_steps if hasattr(opt, 'diffusion_steps') else 3))

    def img2feats(self, img_rgb):
        x = self.base(img_rgb)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        if self.use_diffusion:
            y[-1] = self.diffusion(y[-1])

        return [y[-1]]

    def imgpre2feats(self, vi_img, ir_img, pre_vi_img=None, pre_ir_img=None, pre_hm=None):
        x = self.base(vi_img, ir_img, pre_vi_img, pre_ir_img,  pre_hm)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        if self.use_diffusion:
            y[-1] = self.diffusion(y[-1])

        return [y[-1]]

class Tracker(object):
    def __init__(self, opt):
        self.opt = opt
        self.reset()

    def init_track(self, results):
        for item in results:
            if item['score'] > self.opt.new_thresh:
                self.id_count += 1
                item['active'] = 1
                item['age'] = 1
                item['tracking_id'] = self.id_count
                if not ('ct' in item):
                    bbox = item['bbox']
                    item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                item['velocity'] = [0, 0]
                item['prev_ct'] = item['ct']
                self.tracks.append(item)

    def reset(self):
        self.id_count = 0
        self.tracks = []

    def update_velocity(self, track):
        if 'prev_ct' in track and 'ct' in track:
            curr_velocity = [
                track['ct'][0] - track['prev_ct'][0],
                track['ct'][1] - track['prev_ct'][1]
            ]

            if 'velocity' in track:
                track['velocity'] = [
                    0.7 * curr_velocity[0] + 0.3 * track['velocity'][0],
                    0.7 * curr_velocity[1] + 0.3 * track['velocity'][1]
                ]
            else:
                track['velocity'] = curr_velocity

            track['prev_ct'] = track['ct']
        return track

    def step(self, results, public_det=None):
        results = sorted(results, key=lambda x: x['score'], reverse=True)

        confidence_thresholds = []
        high_thresh = self.opt.new_thresh
        low_thresh = max(0.05, self.opt.new_thresh - 0.4)

        step = (high_thresh - low_thresh) / 5
        for i in range(6):
            confidence_thresholds.append(high_thresh - i * step)

        unmatched_tracks = list(range(len(self.tracks)))
        matched_tracks = []

        for track in self.tracks:
            self.update_velocity(track)

            if 'velocity' in track and track['age'] > 1:
                predicted_ct = [
                    track['ct'][0] + track['velocity'][0],
                    track['ct'][1] + track['velocity'][1]
                ]

                track['original_ct'] = track['ct']
                track['ct'] = predicted_ct

                bbox_width = track['bbox'][2] - track['bbox'][0]
                bbox_height = track['bbox'][3] - track['bbox'][1]
                track['bbox'] = [
                    predicted_ct[0] - bbox_width/2,
                    predicted_ct[1] - bbox_height/2,
                    predicted_ct[0] + bbox_width/2,
                    predicted_ct[1] + bbox_height/2
                ]

        for threshold in confidence_thresholds:
            current_dets = [det for det in results if det['score'] > threshold]

            if not current_dets:
                continue

            remaining_tracks = [self.tracks[i] for i in unmatched_tracks]

            if not remaining_tracks:
                break

            # according to the tracking tasks, can be finetuned            
            adaptive_iou = 0.7 if threshold > high_thresh - step else \
                         0.6 if threshold > high_thresh - 2*step else \
                         0.5 if threshold > high_thresh - 3*step else \
                         0.4 if threshold > high_thresh - 4*step else 0.3

            matches, unmatched_track_indices, unmatched_det_indices = self._associate_detections_to_tracks(
                current_dets, remaining_tracks,
                iou_threshold=adaptive_iou
            )

            for det_idx, track_idx in matches:
                track = remaining_tracks[track_idx]
                det = current_dets[det_idx]

                det['tracking_id'] = track['tracking_id']
                det['age'] = 1
                det['active'] = track['active'] + 1

                if 'velocity' in track:
                    det['velocity'] = track['velocity']
                    det['prev_ct'] = track['ct']

                matched_tracks.append(det)

                results.remove(det)

            unmatched_tracks = [unmatched_tracks[i] for i in unmatched_track_indices]

        for det in results:
            if det['score'] > self.opt.new_thresh:
                self.id_count += 1
                det['tracking_id'] = self.id_count
                det['age'] = 1
                det['active'] = 1
                det['velocity'] = [0, 0]
                det['prev_ct'] = det['ct']
                matched_tracks.append(det)

        for i in unmatched_tracks:
            track = self.tracks[i]
            if track['age'] < self.opt.max_age + 3:
                track['age'] += 1
                track['active'] = 0

                if 'velocity' in track:
                    v = track['velocity']
                    bbox = track['bbox']
                    ct = track['ct']

                    track['prev_ct'] = ct
                    track['ct'] = [ct[0] + v[0], ct[1] + v[1]]
                    track['bbox'] = [
                        bbox[0] + v[0], bbox[1] + v[1],
                        bbox[2] + v[0], bbox[3] + v[1]
                    ]

                    decay_factor = max(0.5, 1.0 - (track['age'] - 1) * 0.1)
                    track['velocity'] = [v[0] * decay_factor, v[1] * decay_factor]

                matched_tracks.append(track)

        self.tracks = matched_tracks
        return matched_tracks

    def _associate_detections_to_tracks(self, detections, tracks, iou_threshold=None):
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        N = len(detections)
        M = len(tracks)

        dets = np.array([det['ct'] + det['tracking'] for det in detections], np.float32)
        track_size = np.array([((track['bbox'][2] - track['bbox'][0]) * \
                              (track['bbox'][3] - track['bbox'][1])) \
                              for track in tracks], np.float32)
        track_cat = np.array([track['class'] for track in tracks], np.int32)
        item_size = np.array([((item['bbox'][2] - item['bbox'][0]) * \
                             (item['bbox'][3] - item['bbox'][1])) \
                             for item in detections], np.float32)
        item_cat = np.array([item['class'] for item in detections], np.int32)
        tracks_pos = np.array([pre_det['ct'] for pre_det in tracks], np.float32)

        dist = (((tracks_pos.reshape(1, -1, 2) - \
                dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))

        size_diff = np.abs(item_size.reshape(N, 1) - track_size.reshape(1, M)) / \
                   np.maximum(item_size.reshape(N, 1), track_size.reshape(1, M))

        cat_mismatch = (item_cat.reshape(N, 1) != track_cat.reshape(1, M))

        dist = dist + size_diff * 100

        invalid = cat_mismatch
        dist = dist + invalid * 1e18

        max_dist_thresh = 1.5 * track_size.reshape(1, M)
        invalid_dist = (dist > max_dist_thresh)
        dist = dist + invalid_dist * 1e18

        if self.opt.hungarian:
            item_score = np.array([item['score'] for item in detections], np.float32)
            dist[dist > 1e18] = 1e18
            matched_indices = linear_assignment(dist)
        else:
            matched_indices = greedy_assignment(copy.deepcopy(dist))

        unmatched_dets = [d for d in range(dets.shape[0]) \
                        if not (d in matched_indices[:, 0])]
        unmatched_tracks = [d for d in range(tracks_pos.shape[0]) \
                          if not (d in matched_indices[:, 1])]

        matches = []
        if self.opt.hungarian:
            for m in matched_indices:
                if dist[m[0], m[1]] > 1e16:
                    unmatched_dets.append(m[0])
                    unmatched_tracks.append(m[1])
                else:
                    matches.append(m)
            matches = np.array(matches).reshape(-1, 2)
        else:
            matches = matched_indices

        return matches, unmatched_tracks, unmatched_dets
