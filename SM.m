% Define the coefficients of the objective function (note: negate c for maximization)
c = [-3; -2];

% Define the coefficients of the inequality constraints
A = [1, 1; 2, 1];

% Define the right-hand side of the inequality constraints
b = [20; 30];

% Call the SimplexMethod function
[solution, z] = SimplexMethod(c, A, b);

% Display the result
disp('Solution:'), disp(solution(1:2));
disp('Objective Value:'), disp(-z);  % Negate z back to original maximization problem

function [solution, z] = SimplexMethod(c, A, b)
    % Step 1: Initialization
    [m, n] = size(A);  % m = number of constraints, n = number of variables
    A = [A, eye(m)];  % Adding identity matrix to the end of A
    c = [c; zeros(m, 1)];  % Extending c with zeros
    basicVariables = n + (1:m)';  % Indices of basic variables
    nonbasicVariables = (1:n)';  % Indices of nonbasic variables

    while true
        % Step 2: Compute the coefficients of the objective function
        cb = c(basicVariables);  % Basic variable coefficients
        cn = c(nonbasicVariables);  % Non-basic variable coefficients
        B = A(:, basicVariables);  % Basic matrix
        N = A(:, nonbasicVariables);  % Non-basic matrix
        
        solution = zeros(n+m, 1);  % Initialize solution here
        solution(basicVariables) = B \ b;

        pi = B' \ cb;  % Price vector
        reducedCosts = cn - N' * pi;
        % Step 3: Optimality check
        if all(reducedCosts >= 0)
            % Optimal solution found
            solution = zeros(n+m, 1);
            solution(basicVariables) = B \ b;
            z = cb' * (B \ b);  % Objective value
            return;
        end

        % Step 4: Choose entering variable
        [~, enterIdx] = min(reducedCosts);  % Choosing most negative reduced cost
        enteringVariable = nonbasicVariables(enterIdx);

        % Step 5: Compute direction of ascent
        d = zeros(n+m, 1);
        d(enteringVariable) = 1;
        d(basicVariables) = -B \ (N(:, enterIdx));

        % Step 6: Determine leaving variable
        theta = inf;
        leavingVariable = -1;
        for i = 1:m
            idx = basicVariables(i);
            if d(idx) < 0
                val = -solution(idx) / d(idx);
                if val < theta
                    theta = val;
                    leavingVariable = idx;
                end
            end
        end

        if leavingVariable == -1
            error('Problem is unbounded.');
        end

        % Step 7: Update basis
        leavingIdx = find(basicVariables == leavingVariable);
        basicVariables(leavingIdx) = enteringVariable;
        nonbasicVariables(enterIdx) = leavingVariable;
    end
end