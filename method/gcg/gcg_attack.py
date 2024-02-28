import gc

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.nn.functional as F

from method import AttackPrompt, MultiPromptAttack, PromptManager
from method import get_embedding_matrix, get_embeddings


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice, ban_token):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], 
            input_embeds, 
            embeds[:,input_slice.stop:,:]
        ], 
        dim=1)
    
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss_1 = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)

    # loss = loss_1
    ban_output = torch.LongTensor(ban_token)[:min(loss_slice.stop - loss_slice.start, len(ban_token))].unsqueeze(0).to(logits.device)

    loss = loss_1 + nn.CrossEntropyLoss()(logits[0,loss_slice.start:loss_slice.start + ban_output.shape[-1],:], ban_output[0]) * -0.1
    
    loss.backward()
    
    return one_hot.grad.clone()

def token_gradients_pure(model, input_ids, input_slice, target_slice, loss_slice, ban_token):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], 
            input_embeds, 
            embeds[:,input_slice.stop:,:]
        ], 
        dim=1)
    
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)

    # loss = loss_1
    
    loss.backward()
    
    return one_hot.grad.clone()


def token_gradients_reward(model, reward_model, input_ids, control_slice, assistant_role_slice, loss_slice):
    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[control_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[control_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:control_slice.start,:], 
            input_embeds, 
            embeds[:,control_slice.stop:,:]
        ], 
        dim=1)
    
    # prepare for reward model input (prefix part)
    embed_weights_reward = reward_model.get_input_embeddings().weight
    basic_input = input_ids[:control_slice.start]
    assistant_role = input_ids[assistant_role_slice.start:assistant_role_slice.stop]
    reward_prefix = torch.cat([basic_input, assistant_role], dim=0)

    one_hot_output = torch.zeros(
        reward_prefix.shape[0],
        embed_weights_reward.shape[0],
        device=reward_model.device,
        dtype=embed_weights_reward.dtype
    )

    one_hot_output.scatter_(
        1, 
        reward_prefix.unsqueeze(1),
        torch.ones(one_hot_output.shape[0], 1, device=reward_model.device, dtype=embed_weights_reward.dtype)
    )
    

    # prepare for reward model input (suffix part)
    logits = model(inputs_embeds=full_embeds).logits
    output_sample = gumbel_softmax(logits[0,loss_slice,:]).to(reward_model.device)
    final_input_one_hot = torch.cat([one_hot_output.detach(), output_sample], dim=0)

    # print(torch.argmax(final_input_one_hot, dim=-1))

    final_input_imbed = (final_input_one_hot.half() @ embed_weights).unsqueeze(0)

    # reward_input_ids = torch.cat([basic_input, assistant_role, output_sample], dim=0).unsqueeze(0).to(model.device)
    loss = reward_model.forward_for_train(inputs_embeds=final_input_imbed).end_rewards.flatten()
    # .end_rewards.flatten()
    loss = loss.mean()
    # print("loss ===================>", loss.item(), torch.argmax(final_input_one_hot, dim=-1))
    # targets = input_ids[target_slice]
    # loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)
    loss.backward()
    
    return one_hot.grad.clone()


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


class GCGAttackPrompt(AttackPrompt):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
    def grad(self, model):
        return token_gradients(
            model, 
            self.input_ids.to(model.device), 
            self._control_slice, 
            self._target_slice, 
            self._loss_slice,
            self.ban_token
        )

    def grad_pure(self, model):
        return token_gradients_pure(
            model, 
            self.input_ids.to(model.device), 
            self._control_slice, 
            self._target_slice, 
            self._loss_slice,
        )

    def grad_reward(self, model, reward_model):
        return token_gradients_reward(
            model,
            reward_model, 
            self.input_ids.to(model.device), 
            self._control_slice, 
            self._assistant_role_slice, 
            self._loss_slice,
        )

class GCGPromptManager(PromptManager):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def sample_control(self, grad, batch_size, topk=256, temp=1, allow_non_ascii=True):

        if not allow_non_ascii:
            grad[:, self._nonascii_toks.to(grad.device)] = np.infty
        top_indices = (-grad).topk(topk, dim=1).indices
        control_toks = self.control_toks.to(grad.device)
        original_control_toks = control_toks.repeat(batch_size, 1)
        new_token_pos = torch.arange(
            0, 
            len(control_toks), 
            len(control_toks) / batch_size,
            device=grad.device
        ).type(torch.int64)
        new_token_val = torch.gather(
            top_indices[new_token_pos], 1, 
            torch.randint(0, topk, (batch_size, 1),
            device=grad.device)
        )
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        return new_control_toks

class GCGMultiPromptAttack(MultiPromptAttack):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def step(self, 
             batch_size=1024, 
             topk=256, 
             temp=1, 
             allow_non_ascii=True, 
             target_weight=1, 
             control_weight=0, 
             verbose=False, 
             opt_only=False,
             filter_cand=True):

        
        # GCG currently does not support optimization_only mode, 
        # so opt_only does not change the inner loop.
        opt_only = False

        main_device = self.models[0].device
        control_cands = []

        for j, worker in enumerate(self.workers):
            worker(self.prompts[j], "grad", worker.model)

        # Aggregate gradients
        grad = None
        for j, worker in enumerate(self.workers):
            new_grad = worker.results.get().to(main_device)
            new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
            if grad is None:
                grad = torch.zeros_like(new_grad)
            if grad.shape != new_grad.shape:
                with torch.no_grad():
                    control_cand = self.prompts[j-1].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
                    control_cands.append(self.get_filtered_cands(j-1, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
                grad = new_grad
            else:
                grad += new_grad

        with torch.no_grad():
            control_cand = self.prompts[j].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
            control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
        del grad, control_cand ; gc.collect()
        
        # Search
        loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
        with torch.no_grad():
            for j, cand in enumerate(control_cands):
                # Looping through the prompts at this level is less elegant, but
                # we can manage VRAM better this way
                progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else enumerate(self.prompts[0])
                for i in progress:
                    for k, worker in enumerate(self.workers):
                        worker(self.prompts[k][i], "logits", worker.model, cand, return_ids=True)

                    logits, ids = zip(*[worker.results.get() for worker in self.workers])
                    loss[j*batch_size:(j+1)*batch_size] += sum([
                        target_weight*self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device) 
                        for www, (logit, id) in enumerate(zip(logits, ids))
                    ])
                    
                    if control_weight == 0:
                        loss[j*batch_size:(j+1)*batch_size] += sum([
                            0.1*self.prompts[k][i].banned_loss(logit, id).mean(dim=-1).to(main_device)
                            for k, (logit, id) in enumerate(zip(logits, ids))
                        ])
                    del logits, ids ; gc.collect()
                    
                    if verbose:
                        progress.set_description(f"loss={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")

            min_idx = loss.argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size
            next_control, cand_loss = control_cands[model_idx][batch_idx], loss[min_idx]
        
        del control_cands, loss ; gc.collect()

        print('Current length:', len(self.workers[0].tokenizer(next_control).input_ids[1:]))
        print(next_control)

        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers)

class GCGMultiPromptAttack_Mine(MultiPromptAttack):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def step(self, 
             batch_size=1024, 
             topk=256, 
             temp=1, 
             allow_non_ascii=True, 
             target_weight=1, 
             control_weight=0, 
             verbose=False, 
             opt_only=False,
             filter_cand=True):

        # GCG currently does not support optimization_only mode, 
        # so opt_only does not change the inner loop.
        opt_only = False

        main_device = self.models[0].device
        control_cands = []

        for j, worker in enumerate(self.workers):
            if self.tylor_work_mode == "grad":
                worker(self.prompts[j], "grad", worker.model)
            elif self.tylor_work_mode == "grad_pure":
                worker(self.prompts[j], "grad_pure", worker.model)
            elif self.tylor_work_mode == "grad_reward":
                worker(self.prompts[j], "grad_reward", worker.model, worker.reward_model)
            else:
                raise NotImplementedError("Explore new!")

        # Aggregate gradients
        grad = None
        for j, worker in enumerate(self.workers):
            new_grad = worker.results.get().to(main_device)
            new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
            if grad is None:
                grad = torch.zeros_like(new_grad)
            if grad.shape != new_grad.shape:
                with torch.no_grad():
                    control_cand = self.prompts[j-1].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
                    control_cands.append(self.get_filtered_cands(j-1, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
                grad = new_grad
            else:
                grad += new_grad

        with torch.no_grad():
            control_cand = self.prompts[j].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
            control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
        del grad, control_cand ; gc.collect()
        
        # Search
        loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
        with torch.no_grad():
            for j, cand in enumerate(control_cands):
                # Looping through the prompts at this level is less elegant, but
                # we can manage VRAM better this way
                progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else enumerate(self.prompts[0])
                for i in progress:
                    for k, worker in enumerate(self.workers):
                        # if self.find_work_mode == "logits":
                        #     worker(self.prompts[j], "logits", worker.model, cand)
                        # elif self.find_work_mode == "rewards":
                        worker(self.prompts[k][i], "rewards", worker.model, worker.reward_model, cand)
                        # else:
                        #     raise NotImplementedError("Explore new!")
                    rewards = [worker.results.get() for worker in self.workers]
                    loss[j*batch_size:(j+1)*batch_size] += sum(rewards).to(loss.device)
            
                    if verbose:
                        progress.set_description(f"loss={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")

            min_idx = loss.argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size
            next_control, cand_loss = control_cands[model_idx][batch_idx], loss[min_idx]
        
        del control_cands, loss ; gc.collect()

        print('Current length:', len(self.workers[0].tokenizer(next_control).input_ids[1:]))
        print(next_control)

        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers)
