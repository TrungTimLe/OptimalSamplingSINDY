function dy = Wendling_dx(x, params)
    % estimate_current_derivatives: Estimates the derivatives of the state 
    % variables at the current state using the governing equations of the 
    % Wendling model.
    %
    % Args:
    %   x: A vector of the current state variables [x_0, y_0, x_1, y_1, x_2, y_2, x_3, y_3]
    %   params: A structure containing all necessary parameters for the model
    %
    % Returns:
    %   dy: A vector of estimated derivatives for the current state

    % Unpack parameters for readability
    A = params.A;
    B = params.B;
    G = params.G;
    a = params.a;
    b = params.b;
    g = params.g;
    C1 = params.C1;
    C2 = params.C2;
    C3 = params.C3;
    C4 = params.C4;
    C5 = params.C5;
    C6 = params.C6;
    C7 = params.C7;
    e0 = params.e0;
    r = params.r;
    v0 = params.v0;
    I = params.I; % External input

    % Sigmoid function
    S = @(u) 2 * e0 ./ (1 + exp(r * (v0 - u)));

    % Compute the derivatives using the governing equations
    dy = zeros(8, 1); % Preallocate for speed
    
    % x_0 and y_0
    dy(1) = x(2); % x_dot_0
    dy(2) = A * a * S(C2 * x(3) - C4 * x(5) - C7 * x(7)) - 2 * a * x(2) - a^2 * x(1); % y_dot_0
    
    % x_1 and y_1
    dy(3) = x(4); % x_dot_1
    dy(4) = A * a * (I / C2 + S(C1 * x(1))) - 2 * a * x(4) - a^2 * x(3); % y_dot_1
    
    % x_2 and y_2
    dy(5) = x(6); % x_dot_2
    dy(6) = B * b * S(C3 * x(1)) - 2 * b * x(6) - b^2 * x(5); % y_dot_2
    
    % x_3 and y_3
    dy(7) = x(8); % x_dot_3
    dy(8) = G * g * S(C5 * x(1) - C6 * x(5)) - 2 * g * x(8) - g^2 * x(7); % y_dot_3 
end
