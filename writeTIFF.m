function writeTIFF(timg, filename)
    assert(nargin > 1, 'Usage: writeTIFF(timg, filename)');
    
    [h,w,bpp] = size(timg);
    t = Tiff(filename, 'w'); 
    tagstruct.ImageLength = size(timg, 1); 
    tagstruct.ImageWidth = size(timg, 2); 
    tagstruct.Compression = Tiff.Compression.None; 
    tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP; 
    tagstruct.Photometric = Tiff.Photometric.MinIsBlack; 
    tagstruct.BitsPerSample = 32;
    tagstruct.SamplesPerPixel = bpp; 
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky; 
    t.setTag(tagstruct); 
    t.write(timg); 
    t.close();
end