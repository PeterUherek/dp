
function pca(x, n_comp, whitened)
    -- Center data
    local mean = torch.mean(x, 1)
    local x_m = x - torch.ger(torch.ones(x:size(1)), mean:squeeze())

    -- Calculate Covariance
    local cov = x_m * x_m:t()
    --  cov:div(x:size(1) - 1)

    -- Get eigenvalues and eigenvectors
    local ce, cv = torch.symeig(cov, 'V')
    -- Sort eigenvalues
    local ce, idx = torch.sort(ce, true)
    
    -- Sort eigenvectors
    cv = cv:index(2, idx:long())

    -- Keep only the top
    if n_comp and n_comp < cv:size(2) then
        ce = ce:sub(1, n_comp)
        cv = cv:sub(1, -1, 1, n_comp)
    end

    -- Check if whitened version
    -- vectors are divided by the singular values to
    -- ensure uncorrelated outputs with unit component-wise variances.
    if not whitened then
        ce:add(1e-5):sqrt()
    end

    -- Get inverse
    local inv_ce = ce:clone():pow(-1)

    -- Make it a matrix with diagonal inv_ce
    local inv_diag = torch.diag(inv_ce)
    
    -- Transform to get U
    local u = x_m:t() * cv * inv_diag

    return u
end