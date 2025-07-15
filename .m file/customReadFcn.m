function img = customReadFcn(filename)
    img = imread(filename);
    img = imresize(img, [224 224]);
    if size(img,3) == 1
        img = cat(3, img, img, img);
    end
    if size(img,3) ~= 3
        img = [];
    end
end
