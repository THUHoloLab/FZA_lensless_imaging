function Or = MyAdjointOperatorPropagation(I,H)

FI = fftshift(fft2(fftshift(I)));
% Or = fftshift(ifft2(fftshift(FI.*conj(H))));
Or = fftshift(ifft2(fftshift(FI./H)));

Or = real(Or);
% Or = Or/max(Or(:));

end