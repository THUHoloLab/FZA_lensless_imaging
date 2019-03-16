function I = MyForwardOperatorPropagation(obj,H)

FO = fftshift(fft2(fftshift(obj)));
I = fftshift(ifft2(fftshift(FO.*H)));

% I = 0.5 * (real(I) + sum(obj(:)));
I = real(I);

end
